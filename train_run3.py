"""Run 3: tests the capacity hypothesis flagged as untested in the
2026-07-29 SLM Training Evaluation Report — does a bigger model move past
the plateau the 17M dense model hit, on the same TinyStories data (now with
the EOS-separator bug fixed)? Trained from random init, not upcycled — the
embedding/layer dimensions differ from the 17M model, so weights aren't
directly transferable.

Wall-clock capped in addition to the step-based cosine schedule, so this is
safe to leave running unattended overnight regardless of actual measured
speed — it will save a final checkpoint and stop cleanly at the time limit
even if it hasn't reached max_steps.
"""

import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm

from slm.model import TinyGPT

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

torch.manual_seed(42)


class Config:
    vocab_size = 8000
    block_size = 256
    n_layer = 8       # was 6
    n_head = 8        # was 6
    n_embd = 512       # was 384
    dropout = 0.1


cfg = Config()
batch_size = 32
max_steps = 50000       # matches Run 2's step count for a clean comparison
eval_interval = 250
eval_batches = 50
lr = 3e-4
weight_decay = 0.01
grad_clip = 1.0
TIME_LIMIT_SECONDS = 5.5 * 3600  # hard safety cap — stop cleanly regardless of step progress

Path("checkpoints_run3").mkdir(exist_ok=True)

train_data = np.memmap("data/processed/train.bin", dtype=np.uint16, mode="r")
val_data = np.memmap("data/processed/val.bin", dtype=np.uint16, mode="r")


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - cfg.block_size - 1, (batch_size,))
    x = torch.stack([
        torch.from_numpy(np.array(data[i:i+cfg.block_size], dtype=np.int64))
        for i in ix
    ])
    y = torch.stack([
        torch.from_numpy(np.array(data[i+1:i+cfg.block_size+1], dtype=np.int64))
        for i in ix
    ])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = []
        for _ in range(eval_batches):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


model = TinyGPT(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}")

optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def cosine_lr(step, warmup=200):
    if step < warmup:
        return lr * step / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return 0.1 * lr + 0.9 * lr * 0.5 * (1 + math.cos(math.pi * progress))


start_time = time.time()
last_step = 0

pbar = tqdm(range(max_steps), desc="training-run3")
for step in pbar:
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT_SECONDS:
        print(f"\nTime limit reached ({elapsed/3600:.2f}h) at step {step}. Stopping cleanly.")
        break

    for param_group in optimizer.param_groups:
        param_group["lr"] = cosine_lr(step)

    xb, yb = get_batch("train")
    _, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    pbar.set_postfix(loss=f"{loss.item():.4f}")
    last_step = step

    if step % eval_interval == 0 or step == max_steps - 1:
        losses = estimate_loss(model)
        print(f"\nstep {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        torch.save({
            "model": model.state_dict(),
            "config": cfg.__dict__,
            "step": step,
        }, f"checkpoints_run3/step_{step}.pt")

# Always save a true final checkpoint, whether we hit max_steps or the time cap
final_losses = estimate_loss(model)
print(f"\nFINAL step {last_step}: train loss {final_losses['train']:.4f}, val loss {final_losses['val']:.4f}")
torch.save({
    "model": model.state_dict(),
    "config": cfg.__dict__,
    "step": last_step,
}, f"checkpoints_run3/step_{last_step}_final.pt")

print(f"Done. Total wall clock: {(time.time() - start_time)/3600:.2f}h, steps completed: {last_step + 1}")

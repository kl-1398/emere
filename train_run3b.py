"""Run 3b: continuation of Run 3, per that report's own Next Experiment Plan.
Run 3 was stopped by a wall-clock cap at step 39,210/50,000 while still
improving (no plateau) — this resumes from that exact checkpoint rather than
starting over, since the model's weights are still useful, only the original
50k-step cosine schedule needs picking back up.

Rather than trying to splice back into the original schedule mid-curve, this
uses a fresh short-warmup cosine decay over a new step budget — the standard
"continued pretraining" pattern. Same architecture, same data, same peak LR,
per the report's explicit "single variable to change: step count only."
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

torch.manual_seed(43)  # different seed for the continuation phase's batch sampling


class Config:
    vocab_size = 8000
    block_size = 256
    n_layer = 8
    n_head = 8
    n_embd = 512
    dropout = 0.1


cfg = Config()
batch_size = 32
START_STEP = 39210               # first new global step, continuing from Run 3's stop point
CONTINUE_STEPS = 40000           # new local step budget for this phase's cosine schedule
eval_interval = 250
eval_batches = 50
lr = 3e-4                        # same peak as Run 3 — only step count changes, per the plan
weight_decay = 0.01
grad_clip = 1.0
TIME_LIMIT_SECONDS = 6 * 3600    # generous safety cap; not sleep-constrained this time, but still capped on principle
RESUME_CKPT = "checkpoints_run3/step_39209_final.pt"

Path("checkpoints_run3b").mkdir(exist_ok=True)

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
ckpt = torch.load(RESUME_CKPT, map_location=device)
model.load_state_dict(ckpt["model"])
print(f"Resumed weights from {RESUME_CKPT} (was step {ckpt['step']})")

n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}")

optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def cosine_lr(local_step, warmup=100):
    if local_step < warmup:
        return lr * local_step / warmup
    progress = (local_step - warmup) / max(1, CONTINUE_STEPS - warmup)
    return 0.1 * lr + 0.9 * lr * 0.5 * (1 + math.cos(math.pi * progress))


start_time = time.time()
last_global_step = START_STEP - 1

pbar = tqdm(range(CONTINUE_STEPS), desc="training-run3b")
for local_step in pbar:
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT_SECONDS:
        print(f"\nTime limit reached ({elapsed/3600:.2f}h) at local step {local_step}. Stopping cleanly.")
        break

    global_step = START_STEP + local_step

    for param_group in optimizer.param_groups:
        param_group["lr"] = cosine_lr(local_step)

    xb, yb = get_batch("train")
    _, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    pbar.set_postfix(loss=f"{loss.item():.4f}", gstep=global_step)
    last_global_step = global_step

    if local_step % eval_interval == 0 or local_step == CONTINUE_STEPS - 1:
        losses = estimate_loss(model)
        print(f"\nglobal step {global_step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        torch.save({
            "model": model.state_dict(),
            "config": cfg.__dict__,
            "step": global_step,
        }, f"checkpoints_run3b/step_{global_step}.pt")

final_losses = estimate_loss(model)
print(f"\nFINAL global step {last_global_step}: train loss {final_losses['train']:.4f}, val loss {final_losses['val']:.4f}")
torch.save({
    "model": model.state_dict(),
    "config": cfg.__dict__,
    "step": last_global_step,
}, f"checkpoints_run3b/step_{last_global_step}_final.pt")

print(f"Done. Total wall clock: {(time.time() - start_time)/3600:.2f}h, global step reached: {last_global_step}")

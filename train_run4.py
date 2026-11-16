"""Run 4: a deliberate ~9x capacity jump over Run 3/3b (33.5M -> 305.6M
params), on the same reasoning that took Run 2 -> Run 3 (loss plateauing
well before the corpus is data-exhausted means capacity, not data, is the
bottleneck). Triggered specifically because Run 3c (bounded continuation
from Run 3b's best checkpoint) confirmed the plateau was real rather than
a temporary flat patch.

Random init, not upcycled — every dimension changed (n_layer 8->18, n_head
8->18, n_embd 512->1152), so none of Run 3c's weights are shape-compatible.
Same data/tokenizer/block_size/batch_size/peak-LR as every prior dense run;
only the architecture and step budget are the variables being changed here.

head_dim = n_embd / n_head = 64, preserving the ratio every prior run
(Run 1/2: 384/6, Run 3/3b/3c: 512/8) has used, rather than hitting an
arbitrary round embedding number that breaks it.

Real measured cost on this hardware (M4 Pro, MPS) before launch: ~5.36s/step
(measured with Run 3c concurrently training on the same GPU, so this is a
pessimistic/conservative estimate, not a lower bound) -> ~74h for the full
50,000-step budget. Wall-clock capped well above that estimate so a slower-
than-expected run still saves a true final checkpoint rather than being
killed mid-step by an external timeout.
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

torch.manual_seed(45)  # next in the sequence: 42 (Run1/2/3), 43 (Run3b), 44 (Run3c)


class Config:
    vocab_size = 8000
    block_size = 256
    n_layer = 18      # was 8 (Run 3/3b/3c)
    n_head = 18       # was 8
    n_embd = 1152     # was 512 -- head_dim = 1152/18 = 64, same ratio as every prior run
    dropout = 0.1


cfg = Config()
batch_size = 32
max_steps = 50000
eval_interval = 250
eval_batches = 50
lr = 3e-4
weight_decay = 0.01
grad_clip = 1.0
TIME_LIMIT_SECONDS = 90 * 3600  # ~90h safety cap -- comfortably above the ~74h measured estimate

Path("checkpoints_run4").mkdir(exist_ok=True)

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

pbar = tqdm(range(max_steps), desc="training-run4")
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
        }, f"checkpoints_run4/step_{step}.pt")

final_losses = estimate_loss(model)
print(f"\nFINAL step {last_step}: train loss {final_losses['train']:.4f}, val loss {final_losses['val']:.4f}")
torch.save({
    "model": model.state_dict(),
    "config": cfg.__dict__,
    "step": last_step,
}, f"checkpoints_run4/step_{last_step}_final.pt")

print(f"Done. Total wall clock: {(time.time() - start_time)/3600:.2f}h, steps completed: {last_step + 1}")

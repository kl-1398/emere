import math
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm

from slm.model import Config, TinyGPT

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

torch.manual_seed(42)

cfg = Config()
batch_size = 32
max_steps = 50000
eval_interval = 250
eval_batches = 50
lr = 3e-4
weight_decay = 0.01
grad_clip = 1.0

Path("checkpoints").mkdir(exist_ok=True)

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
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

def cosine_lr(step, warmup=200):
    if step < warmup:
        return lr * step / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return 0.1 * lr + 0.9 * lr * 0.5 * (1 + math.cos(math.pi * progress))

pbar = tqdm(range(max_steps), desc="training")
for step in pbar:
    for param_group in optimizer.param_groups:
        param_group["lr"] = cosine_lr(step)

    xb, yb = get_batch("train")
    _, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    pbar.set_postfix(loss=f"{loss.item():.4f}")

    if step % eval_interval == 0 or step == max_steps - 1:
        losses = estimate_loss(model)
        print(f"\nstep {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        ckpt = {
            "model": model.state_dict(),
            "config": cfg.__dict__,
            "step": step,
        }
        torch.save(ckpt, f"checkpoints/step_{step}.pt")

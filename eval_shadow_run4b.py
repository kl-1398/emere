"""Companion, read-only evaluator for Run 4b. The training process's stdout
is fully buffered (not a tty), so its "global step ... val loss ..." print
lines are stuck in memory and not visible in train_run4b.log for a long
time. Rather than touch the live training process (risking the run), this
script independently loads each new checkpoint as it's saved and computes
its own val loss on the same held-out data.

Runs on MPS (CPU was tried first and was impractically slow for a 305M
model -- 10+ minutes with no result, pegging ~3.5 CPU cores, while a new
checkpoint lands every ~16 min). This does share the GPU with the training
process, but it's forward-only (no backward pass, no optimizer step) and
uses a small batch count specifically to keep each interruption brief --
a light, occasional dip in training throughput, not a real contender for
GPU time. Purely additive otherwise: reads checkpoint files, never writes
near them.
"""

import json
import re
import time
from pathlib import Path

import numpy as np
import torch

from slm.model import TinyGPT

CKPT_DIR = Path("checkpoints_run4b")
OUT_PATH = Path("eval_shadow_run4b.json")
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


class Config:
    vocab_size = 8000
    block_size = 256
    n_layer = 18
    n_head = 18
    n_embd = 1152
    dropout = 0.1


cfg = Config()
batch_size = 32
eval_batches = 20  # trimmed from 50 (the training script's own count) to keep each interruption brief

val_data = np.memmap("data/processed/val.bin", dtype=np.uint16, mode="r")


def get_val_batch():
    ix = torch.randint(len(val_data) - cfg.block_size - 1, (batch_size,))
    x = torch.stack([
        torch.from_numpy(np.array(val_data[i:i+cfg.block_size], dtype=np.int64))
        for i in ix
    ])
    y = torch.stack([
        torch.from_numpy(np.array(val_data[i+1:i+cfg.block_size+1], dtype=np.int64))
        for i in ix
    ])
    return x.to(device), y.to(device)


def eval_checkpoint(path):
    model = TinyGPT(cfg).to(device)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(eval_batches):
            xb, yb = get_val_batch()
            _, loss = model(xb, yb)
            losses.append(loss.item())
    return sum(losses) / len(losses)


def load_history():
    if OUT_PATH.exists():
        return json.loads(OUT_PATH.read_text())
    return []


def save_history(history):
    OUT_PATH.write_text(json.dumps(history, indent=2))


print(f"Shadow evaluator watching {CKPT_DIR}/ (CPU only, read-only, safe to stop any time)")
history = load_history()
done_steps = {h["step"] for h in history}

while True:
    ckpts = sorted(
        CKPT_DIR.glob("step_*.pt"),
        key=lambda p: int(re.search(r"step_(\d+)", p.stem).group(1)),
    )
    for p in ckpts:
        step = int(re.search(r"step_(\d+)", p.stem).group(1))
        if step in done_steps:
            continue
        try:
            val_loss = eval_checkpoint(p)
        except Exception as e:
            print(f"Skipping {p.name}: {e}")
            continue
        entry = {"step": step, "val_loss": round(val_loss, 4), "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        history.append(entry)
        done_steps.add(step)
        history.sort(key=lambda h: h["step"])
        save_history(history)
        print(f"step {step}: shadow val loss {val_loss:.4f}")

    time.sleep(15)

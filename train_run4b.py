"""Run 4b: bounded continuation from Run 4's BEST checkpoint (step_46750,
val loss 1.3006) rather than its final one (step_49999, val loss 1.3156) --
Run 4's own evaluation report found the last ~3,250 steps had flattened
into an oscillating plateau, so continuing from the final checkpoint would
mean resuming from a worse point than is available.

Purpose: the confirming experiment from that report's Next Experiment Plan,
mirroring the exact Run 3b -> Run 3c pattern at this new 305.6M scale. If
val loss beats 1.3006 by a real margin (>0.01), the plateau wasn't real and
more training at this scale is still worth it. If it doesn't, by the
halfway point of this run's budget, that's the rollback signal to scale the
model up next instead (same logic as Run 3b -> Run 3c and Run 2 -> Run 3).

Same architecture, same data, same peak LR as Run 4 -- single variable
changed is step count, continuing the "continued pretraining" pattern.
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

torch.manual_seed(46)  # next in the sequence: 42 (Run1/2/3), 43 (Run3b), 44 (Run3c), 45 (Run4)


class Config:
    vocab_size = 8000
    block_size = 256
    n_layer = 18
    n_head = 18
    n_embd = 1152
    dropout = 0.1


cfg = Config()
batch_size = 32
START_STEP = 46751               # first new global step, continuing from the BEST checkpoint (not the final one)
CONTINUE_STEPS = 25000           # new local step budget for this phase's cosine schedule
eval_interval = 250
eval_batches = 50
lr = 3e-4                        # same peak as Run 4 -- only step count changes
weight_decay = 0.01
grad_clip = 1.0
TIME_LIMIT_SECONDS = 40 * 3600   # safety cap; ~25,000 steps at Run 4's ~4.02s/step is ~27.9h, generous buffer
RESUME_CKPT = "checkpoints_run4/step_46750.pt"
ROLLBACK_CHECK_STEP = CONTINUE_STEPS // 2   # halfway point, per the report's rollback condition
BEST_VAL_TO_BEAT = 1.3006

Path("checkpoints_run4b").mkdir(exist_ok=True)

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
best_val_seen = None

pbar = tqdm(range(CONTINUE_STEPS), desc="training-run4b")
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
        if best_val_seen is None or losses["val"] < best_val_seen:
            best_val_seen = losses["val"]
        print(f"\nglobal step {global_step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} (best this run: {best_val_seen:.4f})")
        torch.save({
            "model": model.state_dict(),
            "config": cfg.__dict__,
            "step": global_step,
        }, f"checkpoints_run4b/step_{global_step}.pt")

    if local_step == ROLLBACK_CHECK_STEP:
        margin = BEST_VAL_TO_BEAT - (best_val_seen if best_val_seen is not None else 999)
        print(f"\n[Rollback check @ local step {local_step}] best val this run: {best_val_seen:.4f}, "
              f"target to beat: {BEST_VAL_TO_BEAT}, margin: {margin:.4f} "
              f"({'ON TRACK' if margin > 0 else 'NOT YET BEATEN -- watch closely'})")

final_losses = estimate_loss(model)
print(f"\nFINAL global step {last_global_step}: train loss {final_losses['train']:.4f}, val loss {final_losses['val']:.4f}")
torch.save({
    "model": model.state_dict(),
    "config": cfg.__dict__,
    "step": last_global_step,
}, f"checkpoints_run4b/step_{last_global_step}_final.pt")

print(f"Done. Total wall clock: {(time.time() - start_time)/3600:.2f}h, global step reached: {last_global_step}")
print(f"Best val loss this run: {best_val_seen:.4f} vs. Run 4's best of {BEST_VAL_TO_BEAT} "
      f"({'IMPROVED' if best_val_seen < BEST_VAL_TO_BEAT else 'DID NOT IMPROVE'})")

"""Stage 2: upcycles the finished dense TinyGPT checkpoint into an 8-expert
Mixture-of-Experts model (one expert per vault lobe + Private), then fine-
tunes on vault_raw data. Router is supervised on the true source lobe of
each training example, so expert identity stays pinned to lobe identity
rather than the router inventing its own grouping.

Lower steps and lower LR than the base run on purpose: this stage is
specializing a small amount of data into a model that already has general
fluency from stage 1, not learning language from scratch.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm

from slm.moe_model import LOBES, MoEConfig, TinyMoEGPT

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

torch.manual_seed(42)

cfg = MoEConfig()
batch_size = 32
max_steps = 3000
eval_interval = 100
eval_batches = 20
lr = 1e-4  # lower than stage 1 (3e-4) — fine-tuning, not training from scratch
weight_decay = 0.01
grad_clip = 1.0
DENSE_CKPT = "checkpoints/step_49999.pt"

Path("checkpoints_moe").mkdir(exist_ok=True)

with open("data/vault_processed/meta.json") as f:
    meta = json.load(f)

lobe_data = {}
usable_lobes = []
for entry in meta["lobes"]:
    lobe, expert_id, n_tokens = entry["lobe"], entry["expert_id"], entry["tokens"]
    if n_tokens > cfg.block_size + 1:
        slug = lobe.lower().replace(" ", "_")
        arr = np.memmap(f"data/vault_processed/{slug}.bin", dtype=np.uint16, mode="r")
        lobe_data[expert_id] = arr
        usable_lobes.append(expert_id)
        print(f"  usable: {lobe} (expert {expert_id}), {n_tokens:,} tokens")
    else:
        print(f"  SKIPPED (too little data to sample a block): {lobe}")

assert usable_lobes, "no lobe has enough tokens to train on"


def get_batch():
    xs, ys, lobe_ids = [], [], []
    for _ in range(batch_size):
        eid = usable_lobes[torch.randint(len(usable_lobes), (1,)).item()]
        data = lobe_data[eid]
        i = torch.randint(len(data) - cfg.block_size - 1, (1,)).item()
        xs.append(torch.from_numpy(np.array(data[i:i + cfg.block_size], dtype=np.int64)))
        ys.append(torch.from_numpy(np.array(data[i + 1:i + cfg.block_size + 1], dtype=np.int64)))
        lobe_ids.append(eid)
    x = torch.stack(xs).to(device)
    y = torch.stack(ys).to(device)
    lobes = torch.tensor(lobe_ids, dtype=torch.long, device=device)
    return x, y, lobes


@torch.no_grad()
def estimate_loss(model):
    model.eval()
    losses = []
    for _ in range(eval_batches):
        xb, yb, lb = get_batch()
        _, loss = model(xb, yb, lb)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def upcycle_from_dense(model, ckpt_path):
    """Copies the dense checkpoint's shared layers (embeddings, attention,
    layer norms, LM head) directly, and copies its single MLP into every
    expert as a warm start instead of random init."""
    dense = torch.load(ckpt_path, map_location=device)["model"]

    own = model.state_dict()
    own["tok_emb.weight"] = dense["tok_emb.weight"]
    own["pos_emb.weight"] = dense["pos_emb.weight"]
    own["ln_f.weight"] = dense["ln_f.weight"]
    own["ln_f.bias"] = dense["ln_f.bias"]
    own["lm_head.weight"] = dense["lm_head.weight"]

    for i in range(cfg.n_layer):
        for key in ["ln_1.weight", "ln_1.bias", "ln_2.weight", "ln_2.bias",
                    "attn.c_attn.weight", "attn.c_attn.bias",
                    "attn.c_proj.weight", "attn.c_proj.bias"]:
            own[f"blocks.{i}.{key}"] = dense[f"blocks.{i}.{key}"]

        mlp_state = {
            "net.0.weight": dense[f"blocks.{i}.mlp.net.0.weight"],
            "net.0.bias": dense[f"blocks.{i}.mlp.net.0.bias"],
            "net.2.weight": dense[f"blocks.{i}.mlp.net.2.weight"],
            "net.2.bias": dense[f"blocks.{i}.mlp.net.2.bias"],
        }
        for e in range(cfg.n_experts):
            for k, v in mlp_state.items():
                own[f"blocks.{i}.experts.{e}.{k}"] = v.clone()
        # router (blocks.i.router.*) is left at its random init — nothing to upcycle

    model.load_state_dict(own)
    print(f"Upcycled shared layers + {cfg.n_experts} experts from {ckpt_path}")


model = TinyMoEGPT(cfg).to(device)
upcycle_from_dense(model, DENSE_CKPT)

optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def cosine_lr(step, warmup=100):
    if step < warmup:
        return lr * step / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return 0.1 * lr + 0.9 * lr * 0.5 * (1 + math.cos(math.pi * progress))


pbar = tqdm(range(max_steps), desc="training-moe")
for step in pbar:
    for pg in optimizer.param_groups:
        pg["lr"] = cosine_lr(step)

    xb, yb, lb = get_batch()
    _, loss = model(xb, yb, lb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    pbar.set_postfix(loss=f"{loss.item():.4f}")

    if step % eval_interval == 0 or step == max_steps - 1:
        val_loss = estimate_loss(model)
        print(f"\nstep {step}: loss {loss.item():.4f}, eval loss {val_loss:.4f}")
        torch.save({
            "model": model.state_dict(),
            "config": cfg.__dict__,
            "step": step,
        }, f"checkpoints_moe/step_{step}.pt")

print("Done.")

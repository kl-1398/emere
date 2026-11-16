# Training a Small Language Model From Scratch — Emere1.0 Runbook

**Purpose of this document.** This is a complete, self-contained record of how to design, build, and train a genuine from-scratch Small Language Model (SLM) — own transformer architecture, own tokenizer, no pretrained weights, no fine-tuning of someone else's model — using only open tools and your own hardware. It exists so that if you want to do this again in the future without relying on an AI system to guide you, this file alone is enough: goal, reasoning, exact commands, exact code, the mistakes made the first time, and the results achieved. Every script in this document is either the verbatim code that was actually run (Stages 1–6, the MoE extension) or a clearly-labeled generalized template you adapt yourself (the "swap the data source" section).

**Project name:** Emere1.0. **First built:** 2026-07-25 to 2026-07-30, on a MacBook Pro M4 Pro (48GB unified memory, 14 cores). **Framework:** PyTorch. **Tokenizer:** SentencePiece (BPE). **Architecture:** a GPT-style decoder-only transformer, written from scratch (no `transformers` library, no pretrained checkpoint).

---

## 1. Why this exists (read this before starting)

The explicit goal was: **"not piggy-back off another model, but absolutely from scratch — the structure, the data, the lot."** That means:

- No downloading a pretrained model (no GPT-2 weights, no Llama weights, nothing from Hugging Face's model hub).
- No fine-tuning — the model's weights start from random initialization and learn everything from the training data you give it.
- Own tokenizer, trained on your own corpus (not reusing GPT-2's or anyone else's vocabulary).
- Own architecture code, written and understood line by line (not `import transformers; model = GPT2()`).

The only external things used are: a deep learning framework (PyTorch, does linear algebra and autograd — it doesn't know any language), a tokenizer training library (SentencePiece, a statistical algorithm — it doesn't know any language either), and a dataset (raw text — you choose what).

**This document itself is proof the process can be repeated by reading a file, not by asking an AI to remember for you.**

---

## 2. Prerequisites

| Requirement | What was used | Notes |
|---|---|---|
| OS | macOS (Darwin) | Same steps work on Linux; Windows works via WSL2 |
| Python | 3.11.3 | Any 3.10+ should work |
| GPU/accelerator | Apple M4 Pro, PyTorch MPS backend | Works CPU-only too, just much slower. Works on NVIDIA CUDA GPUs with zero code changes — device is auto-detected (Section 6.4) |
| RAM | 48GB unified memory | The dense 17M-param model comfortably trains with far less; 8GB+ is fine for the base scale in this doc |
| Disk | ~3GB for the TinyStories example corpus + tokenized data + checkpoints | Scales with whatever corpus you choose |

**Do not build or train on a network drive / NAS share.** The first attempt at this project was scaffolded directly on a NAS mount, and importing `torch` from there took minutes per run (reading compiled libraries over SMB). Always do the actual coding and training on local disk. A NAS/vault is a fine place to keep a *copy* of the finished code, data, and checkpoints as a durable record — just don't execute from it.

---

## 3. Environment setup

```bash
# 1. Create the project folder
mkdir -p ~/slm-project && cd ~/slm-project

# 2. Create and activate a virtual environment (Python 3.11+)
python3 -m venv .venv
source .venv/bin/activate

# 3. Install exact dependencies (these are the pinned versions that were
#    verified working together on 2026-07-25 — if you're doing this a long
#    time later, these exact versions may no longer be installable; try
#    the unpinned form first and pin only if something breaks)
pip install torch datasets sentencepiece numpy tqdm huggingface_hub

# Versions actually used, for reference / troubleshooting:
#   torch==2.13.0
#   datasets==5.0.0
#   sentencepiece==0.2.2
#   numpy==2.4.6
#   tqdm==4.69.1
#   huggingface_hub==1.24.0
```

Create the directory scaffold:

```bash
mkdir -p data/raw data/processed tokenizer checkpoints slm
touch slm/__init__.py
```

Final structure before writing any code:

```
slm-project/
├── .venv/
├── data/
│   ├── raw/            # plain text, written by prepare_data.py
│   └── processed/      # tokenized binaries, written by encode_data.py
├── tokenizer/          # trained SentencePiece model, written by train_tokenizer.py
├── checkpoints/        # saved model weights during training
├── slm/
│   ├── __init__.py
│   └── model.py        # the transformer architecture
├── prepare_data.py
├── train_tokenizer.py
├── encode_data.py
├── train.py
└── sample.py
```

---

## 4. Pipeline overview

```
 [choose a data source] --> prepare_data.py --> data/raw/{train,val}.txt
                                                        |
                                                        v
                                          train_tokenizer.py (once)
                                                        |
                                                        v
                                          tokenizer/*.model, *.vocab
                                                        |
                                                        v
                              encode_data.py --> data/processed/{train,val}.bin
                                                        |
                                                        v
                                    train.py  (random-init model, learns from .bin)
                                                        |
                                                        v
                                      checkpoints/step_N.pt (repeated every eval_interval)
                                                        |
                                                        v
                                          sample.py (generate text from a checkpoint)
```

Everything downstream of "choose a data source" is generic — it does not care whether the text is storybooks, dictionary definitions, or math equations. That is the whole point of Section 5 below.

---

## 5. Stage 1 — Data acquisition (the swappable part)

**Contract every data source must satisfy:** produce `data/raw/train.txt` and `data/raw/val.txt`, plain UTF-8 text, **one training example per line**, no embedded newlines inside an example (strip/replace them), blank lines are fine (they get skipped downstream). That's it. Everything below this line in the pipeline only ever reads those two files.

Below are four working source options. Pick one, or write your own function following the same shape — `prepare_data.py` is genuinely just "produce a list of text strings, then write them one per line."

```python
"""
prepare_data.py — Stage 1: produce data/raw/train.txt and data/raw/val.txt.

Change SOURCE below to pick what the model learns from. Add your own
source_xxx() function if none of these fit — it just needs to return a
list (or generator) of strings.
"""

from pathlib import Path
import random

OUT = Path("data/raw")
OUT.mkdir(parents=True, exist_ok=True)

# ---- CHANGE THIS to pick a data source ----
SOURCE = "storybook"   # "storybook" | "local_folder" | "dictionary" | "math"

VAL_FRACTION = 0.01     # fraction of examples held out for validation
SEED = 42


def write_split(examples, val_fraction=VAL_FRACTION, seed=SEED):
    rng = random.Random(seed)
    examples = [e for e in examples if e and e.strip()]
    rng.shuffle(examples)
    n_val = max(1, int(len(examples) * val_fraction))
    val, train = examples[:n_val], examples[n_val:]

    def clean(s):
        return s.replace("\r\n", " ").replace("\r", " ").replace("\n", " ").strip()

    with open(OUT / "train.txt", "w", encoding="utf-8") as f:
        for ex in train:
            f.write(clean(ex) + "\n")
    with open(OUT / "val.txt", "w", encoding="utf-8") as f:
        for ex in val:
            f.write(clean(ex) + "\n")
    print(f"Wrote {len(train):,} train / {len(val):,} val examples to {OUT}")


# ---------------------------------------------------------------------------
# Option A — Storybook / any Hugging Face text dataset.
# This is the EXACT code Emere1.0 actually used, verbatim, with the
# dataset's own train/validation split used directly (no manual splitting
# needed — TinyStories already ships both).
# ---------------------------------------------------------------------------
def source_storybook_exact_as_used():
    """This is what actually ran — kept separate because it writes both
    files directly using the dataset's own split, rather than the generic
    write_split() above. Run this function's body directly if you want to
    reproduce Emere1.0's exact original data stage."""
    from datasets import load_dataset

    train_ds = load_dataset("roneneldan/TinyStories", split="train")
    val_ds = load_dataset("roneneldan/TinyStories", split="validation")

    def write_txt(ds, path):
        with open(path, "w", encoding="utf-8") as f:
            for row in ds:
                text = row["text"].strip()
                if text:
                    f.write(text.replace("\r\n", "\n").replace("\r", "\n"))
                    f.write("\n")

    write_txt(train_ds, OUT / "train.txt")
    write_txt(val_ds, OUT / "val.txt")


def source_storybook():
    """Generic version, for any other HF text dataset without a ready
    validation split — swap the dataset name and text field as needed."""
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split="train")
    return [row["text"].strip() for row in ds if row["text"].strip()]


# ---------------------------------------------------------------------------
# Option B — Your own folder of text/markdown files. Works for a personal
# notes vault, a folder of public-domain books, scanned+OCR'd documents,
# anything readable as plain text. Splits files on blank lines by default
# so one "chunk" (paragraph/section) becomes one training example.
# ---------------------------------------------------------------------------
def source_local_folder(folder="data/source_docs", pattern="*.txt", chunk_on="\n\n"):
    examples = []
    for path in sorted(Path(folder).rglob(pattern)):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for chunk in text.split(chunk_on):
            chunk = chunk.strip()
            if chunk:
                examples.append(chunk)
    return examples


# ---------------------------------------------------------------------------
# Option C — A dictionary file. Expects a plain-text file where each line
# is already one entry, e.g. "aardvark: a nocturnal burrowing mammal...".
# Sources for a file like this: a Wiktionary or WordNet plain-text export,
# or any offline dictionary dump you already have — this function doesn't
# fetch one for you, it just ingests whatever you point it at.
# ---------------------------------------------------------------------------
def source_dictionary(path="data/source_docs/dictionary.txt"):
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    return [line.strip() for line in lines if line.strip()]


# ---------------------------------------------------------------------------
# Option D — Procedurally generated math. Fully offline, infinitely
# reproducible, zero external download dependency. Good for (a) smoke-
# testing the whole pipeline with no internet required, or (b) actually
# training a model whose "language" is arithmetic. Expand the `ops` list
# or nest expressions for harder problems.
# ---------------------------------------------------------------------------
def source_math(n_examples=500_000, max_value=999, seed=SEED):
    rng = random.Random(seed)
    ops = [
        ("+", lambda a, b: a + b),
        ("-", lambda a, b: a - b),
        ("*", lambda a, b: a * b),
    ]
    examples = []
    for _ in range(n_examples):
        a, b = rng.randint(0, max_value), rng.randint(0, max_value)
        sym, fn = rng.choice(ops)
        examples.append(f"{a} {sym} {b} = {fn(a, b)}")
    return examples


SOURCES = {
    "storybook": source_storybook,
    "local_folder": source_local_folder,
    "dictionary": source_dictionary,
    "math": source_math,
}

if __name__ == "__main__":
    write_split(SOURCES[SOURCE]())
```

Run it:

```bash
python prepare_data.py
```

---

## 6. Stage 2 — Train the tokenizer

A tokenizer turns raw text into a fixed vocabulary of integer IDs the model can actually learn on. It is trained statistically from your corpus — it has to be retrained any time your corpus's *character/word distribution* changes substantially (natural language vs. math symbols vs. code).

**This is the exact code used:**

```python
# train_tokenizer.py
import sentencepiece as spm
from pathlib import Path

Path("tokenizer").mkdir(exist_ok=True)

spm.SentencePieceTrainer.train(
    input="data/raw/train.txt",
    model_prefix="tokenizer/tinyslm",
    vocab_size=8000,
    model_type="bpe",
    character_coverage=1.0,
    byte_fallback=True,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
)

print("Tokenizer written to tokenizer/tinyslm.model and .vocab")
```

```bash
python train_tokenizer.py
```

**How to size `vocab_size` and `model_type` for a different corpus** (rule of thumb, not measured science — start here and adjust if training is unstable or the tokenizer errors out):

| Corpus type | `vocab_size` | `model_type` | Why |
|---|---:|---|---|
| Natural language prose (stories, articles) | 4,000–16,000 | `bpe` | Enough merges to capture common words/subwords without an enormous embedding table |
| Dictionary / word lists | 8,000–16,000 | `bpe` or `unigram` | Similar to prose; slightly higher vocab helps if entries span many domains |
| Math / symbolic / tiny alphabet (digits, `+ - * / =`) | 20–100 | `bpe` (keep `byte_fallback=True`) | The alphabet is tiny; a large vocab_size will fail to train or waste capacity on IDs that never get used |
| Source code | 8,000–32,000 | `bpe` | Needs to capture both natural-language comments and syntax tokens |

`pad_id=0, unk_id=1, bos_id=2, eos_id=3` is an arbitrary but consistent convention — keep it, because `encode_data.py` and the training/sampling scripts assume `eos_id` is reserved and available via `sp.eos_id()`.

---

## 7. Stage 3 — Encode text into token binaries

This turns `data/raw/*.txt` into flat arrays of token IDs on disk (`.bin` files), which is what actually gets fed to the model during training — reading a memory-mapped array of integers is far faster than re-tokenizing text on the fly every step.

**This is the exact code used** — including a bug fix baked directly into it (see Section 12, "Known pitfalls," item 3, for why the `EOS_ID` append matters):

```python
# encode_data.py
from pathlib import Path
import numpy as np
import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file="tokenizer/tinyslm.model")
Path("data/processed").mkdir(parents=True, exist_ok=True)

EOS_ID = sp.eos_id()

def encode_file(infile, outfile, batch_lines=10000):
    # Encoding the whole file as one string overflows SentencePiece's
    # internal length check on large corpora (hit at ~1.9GB) and would also
    # hold hundreds of millions of Python ints in memory at once. Stream it
    # line-by-line (one example per line, per prepare_data.py) in batches.
    total_tokens = 0
    with open(infile, encoding="utf-8") as f_in, open(outfile, "wb") as f_out:
        batch = []
        for line in f_in:
            line = line.rstrip("\n")
            if not line:
                continue
            batch.append(line)
            if len(batch) >= batch_lines:
                ids = sp.encode(batch, out_type=int)
                for seq in ids:
                    arr = np.array(seq + [EOS_ID], dtype=np.uint16)
                    arr.tofile(f_out)
                    total_tokens += len(arr)
                batch = []
        if batch:
            ids = sp.encode(batch, out_type=int)
            for seq in ids:
                arr = np.array(seq + [EOS_ID], dtype=np.uint16)
                arr.tofile(f_out)
                total_tokens += len(arr)
    print(f"{outfile}: {total_tokens:,} tokens")

encode_file("data/raw/train.txt", "data/processed/train.bin")
encode_file("data/raw/val.txt", "data/processed/val.bin")
```

```bash
python encode_data.py
```

Note: `dtype=np.uint16` caps you at 65,535 distinct token IDs — fine for the vocab sizes in the table above. If you ever train a tokenizer with `vocab_size > 65535`, switch this to `np.uint32` (and update the same dtype in `train.py`/`sample.py`).

---

## 8. Stage 4 — The model architecture

This is a from-scratch GPT-style decoder-only transformer: token + positional embeddings, N transformer blocks (self-attention + MLP with residual connections and layer norm), a final layer norm, and a linear head projecting back to vocabulary size. No `transformers` library import anywhere.

**This is the exact code used (`slm/model.py`):**

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Config:
    vocab_size = 8000
    block_size = 256
    n_layer = 6
    n_head = 6
    n_embd = 384
    dropout = 0.1

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx
```

**Config field meaning and how to scale it:**

| Field | Meaning | Emere1.0 values used |
|---|---|---|
| `vocab_size` | Must exactly match the tokenizer's `vocab_size` | 8000 |
| `block_size` | Max context length (tokens) the model can attend over | 256 |
| `n_layer` | Number of transformer blocks stacked | 6 (Run 1/2) → 8 (Run 3/3b) |
| `n_head` | Attention heads per block (`n_embd` must be divisible by this) | 6 → 8 |
| `n_embd` | Embedding/hidden dimension | 384 → 512 |
| `dropout` | Regularization strength | 0.1 |

**Scaling rule of thumb, validated by real results (Section 14):** if validation loss plateaus *before* the model has been through even one full pass ("epoch") of your training data, the bottleneck is model **capacity**, not data — increase `n_layer`/`n_head`/`n_embd`. If loss is still dropping steadily when you run out of data (or only plateaus after several epochs), the bottleneck is **data volume** — get more data before growing the model further. Track "tokens processed ÷ total corpus tokens" every run so you actually know which regime you're in — this was calculated by hand in Emere1.0's case, not logged automatically.

---

## 9. Stage 5 — Training loop

**This is the exact code used (`train.py`), for the base dense model:**

```python
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
```

```bash
python train.py
```

**Hyperparameter meaning:**

| Field | Meaning |
|---|---|
| `batch_size` | Number of sequences processed per optimizer step |
| `max_steps` | Total optimizer steps to run |
| `eval_interval` | How often to compute val loss and save a checkpoint |
| `lr` (learning rate) | Peak learning rate, reached after `warmup` steps, then cosine-decayed to 10% of peak by `max_steps` |
| `weight_decay` | AdamW L2-style regularization |
| `grad_clip` | Clips gradient norm to prevent instability spikes |

**Device auto-detection** (`cuda` → `mps` → `cpu`) means this exact script runs unmodified on an NVIDIA GPU, an Apple Silicon Mac, or CPU-only — that's why zero code changes were needed when this project later moved to an NVIDIA Jetson (CUDA) box.

**For a long unattended run, add a wall-clock safety cap** (this is what Run 3/3b actually used, see Section 13.2 for the full script) — wrap the loop with an elapsed-time check that breaks cleanly and always saves a final checkpoint, so an overnight run can never overshoot however much time you have available regardless of whether your steps-per-second estimate was right.

---

## 10. Stage 6 — Sampling (generate text from a trained checkpoint)

**This is the exact code used (`sample.py`):**

```python
import sys
import torch
import sentencepiece as spm

from slm.model import Config, TinyGPT

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

prompt = sys.argv[1] if len(sys.argv) > 1 else "Once upon a time"
ckpt_path = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/step_4999.pt"

sp = spm.SentencePieceProcessor(model_file="tokenizer/tinyslm.model")

cfg = Config()
model = TinyGPT(cfg).to(device)

ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt["model"])
model.eval()

ids = sp.encode(prompt, out_type=int)
x = torch.tensor([ids], dtype=torch.long, device=device)

with torch.no_grad():
    y = model.generate(x, max_new_tokens=120, temperature=0.9, top_k=40)

out = sp.decode(y[0].tolist())
print(out)
```

```bash
python sample.py "Once upon a time" checkpoints/step_49999.pt
```

Note: `generate()` always produces exactly `max_new_tokens` tokens — it does not stop early at an end-of-sequence token, even though the tokenizer has one (`eos_id`). This is a known cosmetic gap (Section 12, item 5) — fine to leave as-is, or add an early-stop check yourself if you want clean single-example completions.

---

## 11. Optional advanced stage — multi-domain Mixture-of-Experts (MoE)

This stage is **optional** and was a deliberate later extension, not part of the minimum path to a working SLM. Skip straight to Section 12 if you just want one working model. Do this stage if you specifically want the model's *internal structure* to mirror separate content domains (in the original project: 7 personal "brain lobes" + one more domain, each getting its own dedicated sub-network) so you can later force the model to "answer from" one domain specifically.

**Key technical fact that motivates this whole stage:** a trained dense transformer stores knowledge as distributed weights spread across every layer — there is no way to carve an already-trained dense model into labeled regions after the fact. Real modularity has to be designed in before/during training. The two realistic options are (a) real separate sub-networks with a gating router (Mixture-of-Experts — used here), or (b) lightweight adapter modules per domain on a frozen base (LoRA-style — lighter but less "real" separation). This project chose MoE.

**Architecture (`slm/moe_model.py`)** — same attention block as the dense model, but each block's MLP is replaced by N independent `Expert` MLPs plus a small router network that decides, per token, which experts to use:

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Replace this with whatever domains you actually want — these were named
# after a personal "brain lobe" structure, but the mechanism is generic:
# any list of named content domains works the same way.
LOBES = [
    "Brainstem",
    "Cerebellum",
    "Frontal Lobe",
    "Hippocampus",
    "Occipital Lobe",
    "Parietal Lobe",
    "Temporal Lobe",
    "Private",
]


class MoEConfig:
    vocab_size = 8000
    block_size = 256
    n_layer = 6
    n_head = 6
    n_embd = 384
    dropout = 0.1
    n_experts = len(LOBES)
    top_k = 2                  # each token is routed to its top-2 experts
    router_loss_weight = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class Expert(nn.Module):
    """One expert = one domain. Same shape as the dense model's MLP so a
    trained dense checkpoint's MLP weights can be copied in directly
    (sparse upcycling) instead of starting every expert from random init."""
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class MoEBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.router = nn.Linear(config.n_embd, config.n_experts)
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.n_experts)])
        self.top_k = config.top_k

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        h = self.ln_2(x)

        gate_logits = self.router(h)  # (B, T, E)
        gate_probs = F.softmax(gate_logits, dim=-1)
        topk_probs, topk_idx = gate_probs.topk(self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        expert_outs = torch.stack([e(h) for e in self.experts], dim=-2)  # (B, T, E, C)
        idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, -1, expert_outs.size(-1))
        chosen = torch.gather(expert_outs, dim=2, index=idx_exp)  # (B, T, k, C)
        moe_out = (chosen * topk_probs.unsqueeze(-1)).sum(dim=2)

        x = x + moe_out
        return x, gate_logits


class TinyMoEGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([MoEBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None, lobe_labels=None):
        B, T = idx.size()
        pos = torch.arange(0, T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)

        all_gate_logits = []
        for block in self.blocks:
            x, gate_logits = block(x)
            all_gate_logits.append(gate_logits)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = lm_loss
            if lobe_labels is not None:
                # Every token in a sequence shares that sequence's domain
                # label. Supervising the router directly is what keeps
                # expert identity == domain identity, rather than letting
                # the router invent its own grouping.
                router_losses = []
                for gl in all_gate_logits:
                    Bg, Tg, Eg = gl.shape
                    target = lobe_labels.unsqueeze(1).expand(Bg, Tg).reshape(-1)
                    router_losses.append(F.cross_entropy(gl.reshape(-1, Eg), target))
                router_loss = torch.stack(router_losses).mean()
                loss = lm_loss + self.config.router_loss_weight * router_loss

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, force_lobe=None):
        """force_lobe: optional int index into LOBES. Overrides the router
        so every token is routed 100% to that domain's expert — lets you
        literally ask the model to 'speak from domain X'."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self._forward_with_override(idx_cond, force_lobe)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx

    def _forward_with_override(self, idx, force_lobe):
        if force_lobe is None:
            return self.forward(idx)
        B, T = idx.size()
        pos = torch.arange(0, T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = x + block.attn(block.ln_1(x))
            h = block.ln_2(x)
            x = x + block.experts[force_lobe](h)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, None
```

**Prepare per-domain data** — one plain-text file per domain (`prepare_vault_data.py` pattern, generalize the source folder):

```python
from pathlib import Path

SOURCE_ROOT = Path("/path/to/your/domain/folders")   # one subfolder per domain
OUT = Path("data/domain_raw")
OUT.mkdir(parents=True, exist_ok=True)

def slug(name):
    return name.lower().replace(" ", "_")

for domain in LOBES:
    src = SOURCE_ROOT / domain
    files = sorted(src.rglob("*.md")) if src.exists() else []
    out_path = OUT / f"{slug(domain)}.txt"
    if not files:
        out_path.write_text("")
        continue
    chunks = [f.read_text(encoding="utf-8", errors="ignore").strip() for f in files]
    out_path.write_text("\n\n".join(c for c in chunks if c) + "\n", encoding="utf-8")
```

**Encode each domain with the SAME tokenizer already trained** (do not retrain the tokenizer here — per-domain corpora are usually too small to train a tokenizer on their own):

```python
import json
from pathlib import Path
import numpy as np
import sentencepiece as spm

RAW = Path("data/domain_raw")
OUT = Path("data/domain_processed")
OUT.mkdir(parents=True, exist_ok=True)
sp = spm.SentencePieceProcessor(model_file="tokenizer/tinyslm.model")

meta = {"domains": []}
for expert_id, domain in enumerate(LOBES):
    slug = domain.lower().replace(" ", "_")
    text = (RAW / f"{slug}.txt").read_text(encoding="utf-8")
    if not text.strip():
        meta["domains"].append({"domain": domain, "expert_id": expert_id, "tokens": 0})
        continue
    ids = sp.encode(text, out_type=int)
    np.array(ids, dtype=np.uint16).tofile(OUT / f"{slug}.bin")
    meta["domains"].append({"domain": domain, "expert_id": expert_id, "tokens": len(ids)})

json.dump(meta, open(OUT / "meta.json", "w"), indent=2)
```

**Upcycle a finished dense checkpoint into the MoE model** (copy shared layers directly, clone the single trained MLP into every expert as a warm start instead of random init — only the router starts random) then fine-tune at a lower learning rate:

```python
def upcycle_from_dense(model, ckpt_path, cfg, device):
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
    model.load_state_dict(own)
```

Train with a combined loss (`lm_cross_entropy + router_loss_weight * router_cross_entropy`), lower LR than the base run (`1e-4` vs `3e-4` — fine-tuning, not learning from scratch), and **lobe-balanced batches** (sample domains uniformly, not proportional to corpus size, so small domains still get real gradient signal instead of being drowned out).

**Known limitation of this stage:** domains with very little text (a few thousand tokens) will memorize their handful of source documents rather than generalize — this is expected, not a bug, and is fine/desirable if the intent is "recall this specific small set of facts verbatim." It is not fixable by training longer; it's a data-volume ceiling.

---

## 12. Known pitfalls (read before you hit them yourself)

These are real bugs and gotchas hit during the original build, in the order they were found. Reading this section first will save you the debugging time it took the first time.

1. **Script content swapped.** If you're scaffolding files from an external plan/checklist, verify each file actually contains what its name implies before running anything — a data-download script and a tokenizer-training script were accidentally swapped in the original scaffold (both files existed, both had code, just in the wrong files).

2. **Encoding a huge corpus as one SentencePiece call overflows.** Calling `sp.encode()` on an entire multi-hundred-MB/multi-GB string at once hit an internal SentencePiece length-check overflow around ~1.9GB. Fix: stream the file line-by-line in batches (Section 7's `encode_file`, `batch_lines=10000`), never load+encode the whole corpus as one string.

3. **Missing separators between concatenated examples.** The tokenizer reserves an EOS token ID during training (`eos_id=3`), but nothing actually *used* it — training/validation examples were concatenated back-to-back in the `.bin` files with no boundary marker at all. The model would blend the end of one example into the start of the next mid-generation with no signal it had switched. Fix: append `eos_id` after every example when encoding (already baked into Section 7's code). If you build your own encode script from scratch, do not skip this — verify it by decoding a slice of the raw token array and confirming `eos_id` shows up exactly at your example boundaries, don't just assume it worked.

4. **GPU backend not detected.** Checking only `torch.cuda.is_available()` silently runs Apple Silicon Macs on CPU — Apple's GPU backend is `torch.backends.mps`, a separate check. Always chain `cuda → mps → cpu` (Section 9's device-detection block) so the same script picks up whatever accelerator is actually present, on any machine.

5. **`generate()` doesn't stop at end-of-sequence.** It always produces exactly `max_new_tokens` tokens regardless of whether an EOS token was generated along the way. Cosmetic, not a training bug — samples will look like they "run on" into a new, unrelated example after finishing one. Add an early-stop check on `next_idx == eos_id` if you want clean single-example completions.

6. **Sparse upcycling only works if the architecture dimensions match.** You cannot copy a dense checkpoint's weights into a bigger/smaller model (different `n_embd`/`n_layer`/`n_head`) — the tensor shapes won't line up. If you're scaling the model up (Section 8's scaling rule of thumb), that new, bigger model has to start from random initialization; only same-shaped continuations (like Run 3 → Run 3b in Section 14) can resume from a checkpoint directly.

7. **Never train/execute against a network-mounted drive.** Reading a Python virtual environment's compiled libraries (like `torch`) over SMB from a NAS added minutes to every single script launch. Do all real work on local disk; sync to a NAS/vault afterward purely as a backup/record copy.

8. **Cap long unattended runs on wall-clock time, not just step count.** If you don't know your exact steps/second in advance (new hardware, new model size), a step-count-only budget can run far longer than you intended. Add a `time.time()` check inside the loop that breaks cleanly and always saves a true final checkpoint — this is what let Run 3 safely run overnight unattended (Section 14).

---

## 13. Evaluation — do this after every training run, without exception

The discipline that produced comparable, trustworthy results across every run in Section 14: **write a filled-in copy of the same template after every single training run**, dense or MoE, using real numbers pulled from logs/checkpoints/fresh generations — never estimated or remembered. Save it as a new file each time (never overwrite the template or a previous run's report), so Section "Comparison to Previous Runs" below can pull real historical numbers.

**Canonical template (16 sections) — copy this verbatim into a new file per run:**

```markdown
# SLM Training Evaluation Report Template

## 1. Run Summary
- Run name / Date started / Date completed / Status / Objective / Author / Notes

## 2. Environment
- Project folder / Git commit / Python version / Framework versions /
  Hardware / OS / Device used for training / Mixed precision enabled /
  Checkpoint directory

## 3. Dataset
- Dataset name / source / domain / raw size / cleaned size /
  train/val/test split / estimated total tokens / data quality notes /
  license notes

## 4. Tokenizer
- Type / training corpus / vocab size / special tokens / file path /
  encoding format / notes

## 5. Model Configuration
- Architecture type / model class file / parameter count / n_layer /
  n_head / n_embd / context length / dropout / activation function /
  positional encoding / weight tying / initialization notes

## 6. Training Configuration
- Total planned steps / completed steps / batch size / grad accumulation /
  effective batch size / tokens per step / optimizer / learning rate /
  LR schedule / warmup steps / weight decay / grad clipping /
  eval interval / checkpoint interval / random seed / resumed from checkpoint?

## 7. Training Results
| Metric | Value | Notes |
|---|---:|---|
| Final train loss | | |
| Final validation loss | | |
| Best validation loss | | |
| Train perplexity | | |
| Validation perplexity | | |
| Total runtime | | |
| Tokens processed | | |
| Approx. epochs completed | | |
| Best checkpoint | | |

## 8. Loss Curve Assessment
- Steady decrease (train)? / Steady decrease (val)? / Train/val stay close?
  / Any instability or spikes? / Underfitting, good fit, or overfitting? /
  Short interpretation

## 9. Sample Generations (x3)
### Sample N
- Checkpoint used / Prompt / Decoding settings / Output / Assessment

## 10. Qualitative Evaluation Checklist
- Grammar/syntax / Prompt adherence / Consistency / Coherence /
  Repetition / Looping/degeneration / Long-range consistency /
  Domain fit / Overall quality

## 11. Failure Modes
- Observed issues / When they appear / Likely cause / Severity / Examples

## 12. Comparison to Previous Runs
| Run | Steps | Train loss | Val loss | Val perplexity | Notes |
|---|---:|---:|---:|---:|---|
| Current run | | | | | |
| Previous run 1 | | | | | |
| Previous run 2 | | | | | |

## 13. Interpretation
- What improved? / What didn't? / Still learning? / More training justified?
  / Main bottleneck: steps, model size, context length, or data?

## 14. Decision
- Recommended next action / Why / Do not change yet / Checkpoint to keep /
  Checkpoint to discard

## 15. Next Experiment Plan
- Experiment name / Single variable to change / Hypothesis /
  Success criteria / Metrics to monitor / Rollback condition

## 16. Appendix
### Commands used
### File paths
### Extra notes
```

**Non-negotiable rules for filling it in:**
- Every field comes from a real log, checkpoint file, or a freshly-generated sample — never a fabricated or "typical" number.
- If a data point wasn't preserved (e.g., no per-step log was kept), say so plainly rather than guessing.
- Generate all 3 sample-generation entries fresh, specifically for this report — don't reuse a previous run's samples.
- Pull the "Comparison to Previous Runs" numbers from the actual prior reports, not memory.

---

## 14. Worked example — Emere1.0's actual results

This is the real, measured outcome of following this exact process once, on a 450M-token English storybook corpus (TinyStories), M4 Pro / MPS. Use it as your sanity-check baseline for "is my run in the right ballpark," not as a promise your numbers will match — a different corpus (dictionary, math, code) will behave differently.

| Run | Model | Steps | Train loss | Val loss | Key finding |
|---|---|---:|---:|---:|---|
| Run 1 | 17M params (6L/6H/384E) | 5,000 | 2.4933 | 2.4926 | First working model. Coherent short stories. Only ~9.1% of one epoch — nowhere near data-exhausted. |
| Run 2 | Same 17M model | 50,000 | — | 1.7209 | Clearly plateaued by the end — capacity-limited, not data-limited. |
| MoE Stage 3 | 66.5M params, 8 experts, upcycled from Run 2 | 3,000 (fine-tune) | — | (combined loss only, not decomposed) | Routing genuinely tracked domain content; small domains memorized rather than generalized (expected). |
| Run 3 | 33.5M params (8L/8H/512E), random init, EOS bug fixed | 39,210 / 50,000 (time-capped) | — | 1.5715 | Bigger model beat the 17M plateau with fewer steps, and was *still improving* when cut off — confirmed capacity was the bottleneck, not data. |
| Run 3b | Same 33.5M model, resumed from Run 3 | +40,000 (79,209 cumulative) | 1.4766 (final) | **1.4632** (best, at step 73,710 — not the final checkpoint) | Best result achieved. Last ~10k steps showed early flattening — next step would be either more data or another capacity increase. |

**Concrete lessons this progression teaches about scaling:**
- Going from 17M → 33.5M params (roughly 2x) moved the achievable loss from ~1.72 to ~1.57 on the *same* data — capacity was the binding constraint at 17M.
- The best checkpoint is not always the last one saved — always compare `val loss` across all saved checkpoints from a run, don't assume monotonic improvement holds all the way to the end.
- A resumed continuation (Run 3b) used a *fresh* short-warmup cosine schedule over a *new* step budget rather than trying to splice back into the original schedule mid-curve — this is the standard, simpler way to continue training past an interruption.

---

## 15. Full command cheat sheet (copy-paste, start to finish)

```bash
# --- Setup ---
mkdir -p ~/slm-project && cd ~/slm-project
python3 -m venv .venv && source .venv/bin/activate
pip install torch datasets sentencepiece numpy tqdm huggingface_hub
mkdir -p data/raw data/processed tokenizer checkpoints slm
touch slm/__init__.py

# --- Put the code in place ---
# slm/model.py        <- Section 8
# prepare_data.py     <- Section 5 (pick SOURCE)
# train_tokenizer.py  <- Section 6
# encode_data.py      <- Section 7
# train.py            <- Section 9
# sample.py           <- Section 10

# --- Run the pipeline, in order ---
python prepare_data.py          # Stage 1: SOURCE -> data/raw/{train,val}.txt
python train_tokenizer.py       # Stage 2: -> tokenizer/tinyslm.{model,vocab}
python encode_data.py           # Stage 3: -> data/processed/{train,val}.bin
python train.py                 # Stage 4-5: -> checkpoints/step_N.pt (repeats)

# --- Generate from a trained checkpoint ---
python sample.py "Once upon a time" checkpoints/step_49999.pt

# --- After the run finishes: write the evaluation report (Section 13) ---
# Copy the template into a new dated file, fill it in from real numbers.
```

---

## 16. If you're reading this with no memory of building it before

Start at Section 2, follow it top to bottom in order — Section 3 (setup) → Section 5 (pick a data source) → Section 6 (tokenizer) → Section 7 (encode) → Section 8 (architecture) → Section 9 (train) → Section 10 (sample). Section 11 (MoE) is optional and separate. Section 12 exists so you don't re-discover the same bugs. Section 13 is the discipline that keeps every future run comparable to this one. Nothing in this pipeline calls out to an external AI system at any stage — every step is a fixed, deterministic (given a random seed) program you run yourself.

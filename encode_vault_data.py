"""Encodes each lobe's raw text (from prepare_vault_data.py) into a token
.bin file using the existing SentencePiece tokenizer (trained on TinyStories
— reused rather than retrained, since these corpora are far too small to
train a tokenizer on their own). Small enough per-lobe to encode in one call,
no streaming needed."""

import json
from pathlib import Path

import numpy as np
import sentencepiece as spm

from slm.moe_model import LOBES

RAW = Path("data/vault_raw")
OUT = Path("data/vault_processed")
OUT.mkdir(parents=True, exist_ok=True)

sp = spm.SentencePieceProcessor(model_file="tokenizer/tinyslm.model")

meta = {"lobes": []}

for expert_id, lobe in enumerate(LOBES):
    slug = lobe.lower().replace(" ", "_")
    text = (RAW / f"{slug}.txt").read_text(encoding="utf-8")

    if not text.strip():
        meta["lobes"].append({"lobe": lobe, "expert_id": expert_id, "tokens": 0})
        print(f"{lobe}: empty, skipped")
        continue

    ids = sp.encode(text, out_type=int)
    arr = np.array(ids, dtype=np.uint16)
    arr.tofile(OUT / f"{slug}.bin")

    meta["lobes"].append({"lobe": lobe, "expert_id": expert_id, "tokens": len(ids)})
    print(f"{lobe}: {len(ids):,} tokens -> {slug}.bin")

with open(OUT / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\nWrote", OUT / "meta.json")

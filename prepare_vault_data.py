"""Reads each brain-lobe folder (+ Private) from the vault and writes one
plain-text file per lobe. Pure mechanical file I/O — no LLM of any kind
touches this content, in or out. Cerebellum has no markdown files and is
skipped (its expert gets no lobe-specific data, by design, not a bug)."""

from pathlib import Path

VAULT = Path("path/to/your/notes")  # set this to a folder of markdown notes
OUT = Path("data/vault_raw")
OUT.mkdir(parents=True, exist_ok=True)

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


def slug(name):
    return name.lower().replace(" ", "_")


for lobe in LOBES:
    src = VAULT / lobe
    md_files = sorted(src.rglob("*.md")) if src.exists() else []
    out_path = OUT / f"{slug(lobe)}.txt"

    if not md_files:
        out_path.write_text("")
        print(f"{lobe}: no markdown files found, wrote empty file")
        continue

    chunks = []
    for f in md_files:
        text = f.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            chunks.append(text)

    out_path.write_text("\n\n".join(chunks) + "\n", encoding="utf-8")
    total_chars = sum(len(c) for c in chunks)
    print(f"{lobe}: {len(md_files)} files, {total_chars:,} chars -> {out_path}")

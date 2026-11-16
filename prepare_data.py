from datasets import load_dataset
from pathlib import Path

OUT = Path("data/raw")
OUT.mkdir(parents=True, exist_ok=True)

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

print("Wrote:", OUT / "train.txt")
print("Wrote:", OUT / "val.txt")

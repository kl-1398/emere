"""Overnight MoE evaluation: for each usable lobe, force generation through
that lobe's expert; separately, let the router choose freely on the same
prompts. Writes everything to a text file for review in the morning —
no interpretation here, just raw generations for a human (or a later
Claude turn) to read and assess."""

import torch
import sentencepiece as spm

from slm.moe_model import LOBES, MoEConfig, TinyMoEGPT

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

CKPT = "checkpoints_moe/step_2999.pt"
sp = spm.SentencePieceProcessor(model_file="tokenizer/tinyslm.model")

cfg = MoEConfig()
model = TinyMoEGPT(cfg).to(device)
ckpt = torch.load(CKPT, map_location=device)
model.load_state_dict(ckpt["model"])
model.eval()

PROMPTS = [
    "Once upon a time",
    "The plan for today is",
    "I remember when",
    "The most important thing to understand is",
]

usable_lobes = [i for i, lobe in enumerate(LOBES) if lobe != "Cerebellum"]

with open("moe_evaluation_output.txt", "w") as out:
    out.write(f"MoE evaluation — checkpoint {CKPT}\n")
    out.write("=" * 70 + "\n\n")

    for prompt in PROMPTS:
        out.write(f"PROMPT: {prompt!r}\n")
        out.write("-" * 70 + "\n")

        ids = sp.encode(prompt, out_type=int)
        x = torch.tensor([ids], dtype=torch.long, device=device)

        with torch.no_grad():
            y = model.generate(x, max_new_tokens=80, temperature=0.8, top_k=40)
        out.write(f"[router free choice]\n{sp.decode(y[0].tolist())}\n\n")

        for eid in usable_lobes:
            lobe = LOBES[eid]
            with torch.no_grad():
                y = model.generate(x, max_new_tokens=80, temperature=0.8, top_k=40, force_lobe=eid)
            out.write(f"[forced: {lobe}]\n{sp.decode(y[0].tolist())}\n\n")

        out.write("=" * 70 + "\n\n")

print("Wrote moe_evaluation_output.txt")

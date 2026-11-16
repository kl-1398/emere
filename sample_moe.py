"""Generate from the MoE model. Pass a lobe name as the 2nd arg to force
that expert for every token (e.g. 'Private', 'Temporal Lobe') — otherwise
the router picks per-token on its own."""

import sys
import torch
import sentencepiece as spm

from slm.moe_model import LOBES, MoEConfig, TinyMoEGPT

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

prompt = sys.argv[1] if len(sys.argv) > 1 else "Once upon a time"
lobe_arg = sys.argv[2] if len(sys.argv) > 2 else None
ckpt_path = sys.argv[3] if len(sys.argv) > 3 else "checkpoints_moe/step_2999.pt"

force_lobe = LOBES.index(lobe_arg) if lobe_arg else None

sp = spm.SentencePieceProcessor(model_file="tokenizer/tinyslm.model")

cfg = MoEConfig()
model = TinyMoEGPT(cfg).to(device)
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt["model"])
model.eval()

ids = sp.encode(prompt, out_type=int)
x = torch.tensor([ids], dtype=torch.long, device=device)

with torch.no_grad():
    y = model.generate(x, max_new_tokens=120, temperature=0.8, top_k=40, force_lobe=force_lobe)

print(f"[lobe forced: {lobe_arg or 'router choice'}]")
print(sp.decode(y[0].tolist()))

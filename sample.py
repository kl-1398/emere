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

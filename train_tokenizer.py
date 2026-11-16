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
    eos_id=3
)

print("Tokenizer written to tokenizer/tinyslm.model and .vocab")

from pathlib import Path
import numpy as np
import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file="tokenizer/tinyslm.model")
Path("data/processed").mkdir(parents=True, exist_ok=True)

EOS_ID = sp.eos_id()  # reserved as id 3 during tokenizer training, never
                       # actually inserted between stories until this fix —
                       # see SLM Training Evaluation Report, Error Analysis:
                       # this is the root cause of the unmarked story-switch
                       # failure mode observed in both Run 1 and Run 2.

def encode_file(infile, outfile, batch_lines=10000):
    # Encoding the whole file as one string overflows sentencepiece's
    # internal length check on large corpora (hit at ~1.9GB) and would also
    # hold hundreds of millions of Python ints in memory at once. Stream it
    # line-by-line (one story per line, per prepare_data.py) in batches instead.
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
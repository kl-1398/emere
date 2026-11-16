# Emere

**A small language model, trained from scratch on a laptop, with every run written up.**

Emere is a GPT-style decoder-only transformer written in PyTorch. There are no
pre-trained weights: the tokenizer is trained on the corpus, the model starts
from random initialisation, and everything ran on one MacBook Pro (M4 Pro,
48GB, PyTorch MPS).

It was scaled from 16.9M to 305.6M parameters across seven recorded training
runs. Each run has an evaluation report in [`reports/`](reports/), including
the plateaus, the runs that were stopped early, and the result that did not
turn out the way the loss curve suggested.

This is a learning and research project. It is not a usable assistant. It
writes short children's stories, because that is what it was trained on.

## Results

All runs use the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
dataset, a SentencePiece BPE tokenizer (vocab 8,000) and a context of 256 tokens.

| Run | Steps | Parameters | Best val loss | What it showed |
|---|---:|---:|---:|---|
| 1 | 5,000 | 16.9M | 2.4926 | Pipeline works; about 9% of one epoch |
| 2 | 50,000 | 16.9M | 1.7062 | Plateaued by about step 30,000 |
| 3 | 39,210 | 33.5M | 1.5715 | Bigger model helps; stopped by a time cap while still improving |
| 3b | +40,000 | 33.5M | 1.4632 | Best checkpoint was not the final one |
| 3c | +20,000 | 33.5M | 1.4562 | Gain of 0.007, under the 0.01 threshold set in advance: plateau confirmed |
| 4 | 50,000 | 305.6M | 1.3006 | About 9x the capacity, from random initialisation |
| 4b | +25,000 | 305.6M | **1.2707** | Run 4's apparent plateau was not real (perplexity about 3.56) |

Run 3c has no full report; its figures are recorded in the Run 4 and Run 4b reports.

## What I learned

- **Confirm a plateau before paying for a bigger model.** The rule for moving
  from 33.5M to 305.6M parameters was written down before Run 3c started: if
  the improvement was under 0.01, scale up. It came in at 0.007.
- **Resume from the best checkpoint, not the last one.** In Runs 3b and 4 the
  best validation loss was not at the final step.
- **Lower loss did not mean better stories.** Run 4b improved validation loss
  from 1.3006 to 1.2707, but on a three-prompt comparison the samples were not
  uniformly better. That points at the data, not the model size, as the next
  limit. See [report 05](reports/05-run4b-confirming-continuation.md).
- **Known bugs are left documented.** Generation does not stop at
  end-of-sequence, and the model switches story mid-sample because the data
  preparation did not put a separator between stories.

## Mixture of experts

A second model ([`slm/moe_model.py`](slm/moe_model.py)) takes the Run 2
checkpoint and splits it into 8 experts with a learned router, one expert per
section of my personal notes. It was fine-tuned for 3,000 steps on about 658K
tokens of those notes.

The result is a proof of concept for the routing, not a good generator: the
router followed the topic of the prompt instead of collapsing onto the largest
expert, but the smaller experts had too little data and mostly memorised it.

The notes are private. The training data, the checkpoints and the sample
outputs are not in this repository, and the samples section of
[report 06](reports/06-mixture-of-experts.md) is withheld for the same reason.
One section name has been changed to "Private" in the code and reports for
privacy; nothing else about the run is altered.

To try the MoE code, point `prepare_vault_data.py` at your own folder of
markdown notes and edit the `LOBES` list to match your folder names.

## Run it yourself

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python3 prepare_data.py       # downloads TinyStories
python3 train_tokenizer.py    # trains the SentencePiece tokenizer
python3 encode_data.py        # text -> token binaries
python3 train.py              # the 16.9M baseline (Runs 1 and 2)
python3 sample.py             # generate from a checkpoint
```

`train_run3.py` to `train_run4b.py` are the exact scripts used for the later
runs. The continuation scripts (3b, 3c, 4b) expect the earlier run's checkpoint
to exist. The 305.6M runs took over a day each on the hardware above.
[`reports/RUNBOOK.md`](reports/RUNBOOK.md) is a full step-by-step guide to
rebuilding the pipeline from nothing.

Checkpoints are not published (the Run 4 set alone is 232GB).

## Layout

```text
slm/model.py          the transformer (TinyGPT)
slm/moe_model.py      the mixture-of-experts variant
train*.py             one script per run, kept as they were run
prepare_*, encode_*   data preparation
sample*.py, evaluate_moe.py, eval_shadow_run4b.py, watch_run4b.py
scripts/              the shell pipelines used for unattended runs
reports/              one evaluation report per run, plus the runbook
```

## How it was built

Emere is a joint venture between me and Claude Code, and both deserve the
credit. Claude Code bridged my coding knowledge gap; the inspiration and
imagination are mine.

In practice: I decided what to build, set the rules for each run, ran it on my
own hardware and made the calls between runs. Claude Code wrote code with me,
ran some training stages unattended with my approval, and wrote up the
reports. Each report states who did what at the top, and refers to me as
"the Founder".

## License

MIT. See [LICENSE](LICENSE).

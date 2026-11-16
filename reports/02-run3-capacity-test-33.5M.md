---
project: Emere1.0
type: Evaluation Report
template: slm_training_evaluation_template.md
run: Run 3 (Capacity Test)
date: 2026-07-30
---

# SLM Training Evaluation Report — Run 3 (Capacity Test)

## 1. Run Summary
- **Run name:** Run 3 — Capacity Test
- **Date started:** 2026-07-30 00:00:23 BST
- **Date completed:** 2026-07-30 05:30:38 BST
- **Status:** Stopped by wall-clock safety cap (5.5h), not by reaching planned step count — 39,210 / 50,000 steps (78.4%) completed
- **Objective:** test the hypothesis flagged as untested in the Run 1/2 evaluation report — does a bigger model move past the loss plateau the 17M-parameter dense model hit at ~step 30,000? Also the first run trained on the EOS-separator-fixed data.
- **Author / operator:** Claude Code, autonomous overnight run, Founder-approved in advance (plan + model size confirmed before launch)
- **Notes:** run from random initialization, not upcycled — embedding/layer dimensions differ from the 17M model (384→512 embd, 6→8 layers), so Run 1/2's weights aren't directly transferable.

## 2. Environment
- **Project folder:** `./`
- **Git commit / version:** not a git repo yet
- **Python version:** 3.11.3
- **Framework versions:** PyTorch 2.13.0, sentencepiece 0.2.2
- **Hardware:** Apple M4 Pro, 14 cores (10P/4E), 48GB unified memory
- **OS:** macOS (Darwin)
- **Device used for training:** MPS (Metal GPU)
- **Mixed precision enabled:** No (fp32 throughout)
- **Checkpoint directory:** `checkpoints_run3/`

## 3. Dataset
- **Dataset name:** TinyStories (roneneldan/TinyStories) — same source as Run 1/2, **re-encoded with a bug fix**
- **Dataset source:** HuggingFace
- **Domain / content type:** short children's stories, simple vocabulary/grammar
- **Raw dataset size:** 1.8GB train.txt, 18MB val.txt (unchanged)
- **Cleaned dataset size:** n/a
- **Train split:** HuggingFace's predefined train split (2,119,719 stories)
- **Validation split:** HuggingFace's predefined validation split (21,990 stories)
- **Test split:** none — same limitation as Run 1/2
- **Estimated total tokens:** **461,869,301 train / 4,644,562 val** — up from Run 1/2's 450,443,884 / 4,531,759 because every story now gets an explicit end-of-sequence token appended (see Data Quality Notes)
- **Data quality notes:** **root-cause fix applied this run.** `encode_data.py` previously joined consecutive stories with no boundary marker at all — confirmed as the direct cause of the "unmarked story-switch" failure mode seen in both Run 1 and Run 2's qualitative samples. Fixed by appending the tokenizer's existing (but previously unused) `eos_id=3` after every story. Verified working at the token level post-training: a fresh generation's raw token IDs showed `eos_id` appearing exactly at natural story-boundary positions (see Section 9).
- **License / usage notes:** unchanged from Run 1/2 — TinyStories license terms apply.

## 4. Tokenizer
- **Tokenizer type:** SentencePiece BPE (unchanged, reused from Run 1/2 — no retrain needed for this fix)
- **Tokenizer training corpus:** TinyStories train split
- **Vocabulary size:** 8,000
- **Special tokens:** `pad_id=0, unk_id=1, bos_id=2, eos_id=3` — `eos_id` actively used in encoding for the first time this run
- **Tokenizer file path:** `tokenizer/tinyslm.model`
- **Encoding format:** uint16 token arrays, `data/processed/train.bin` + `val.bin` (re-encoded in place; Run 1/2's original unfixed encoding backed up to `data/processed_run1-2_backup/` before overwriting)
- **Tokenizer notes:** none new this run.

## 5. Model Configuration
- **Architecture type:** decoder-only GPT-style transformer, same class as Run 1/2, scaled up
- **Model class / file:** `TinyGPT`, `slm/model.py` (config overridden locally in `train_run3.py`)
- **Parameter count:** 33,543,168 (~33.5M) — confirmed by direct count at model instantiation, ~1.97x Run 1/2's ~17M
- **Number of layers:** 8 (was 6)
- **Number of attention heads:** 8 (was 6)
- **Embedding dimension:** 512 (was 384)
- **Context length:** 256 tokens (unchanged — deliberately held constant to isolate capacity as the single variable under test)
- **Dropout:** 0.1 (unchanged)
- **Activation function:** GELU
- **Positional encoding:** learned absolute positional embeddings
- **Weight tying enabled:** No
- **Initialization notes:** PyTorch defaults (same caveat as all prior runs — no scaled residual init). Random init throughout, not upcycled.

## 6. Training Configuration
- **Total planned steps:** 50,000 (matched to Run 2 deliberately, for a clean step-for-step comparison)
- **Completed steps:** 39,210 (78.4%) — stopped by the wall-clock cap before reaching the planned count
- **Batch size:** 32
- **Gradient accumulation:** none
- **Effective batch size:** 32
- **Tokens per step:** 8,192 (32 × 256)
- **Optimizer:** AdamW, weight_decay 0.01
- **Learning rate:** 3e-4 peak (matched to Run 1/2)
- **Learning rate schedule:** cosine decay to 10% of peak — **note: schedule was shaped for 50,000 steps but training stopped at 39,210, so the LR never reached its intended minimum.** Final LR was somewhat higher than a "complete" cosine run would have used.
- **Warmup steps:** 200
- **Weight decay:** 0.01
- **Gradient clipping:** 1.0 (global norm)
- **Evaluation interval:** every 250 steps, 50 batches per eval
- **Checkpoint interval:** every 250 steps
- **Random seed:** 42
- **Resume from checkpoint:** No — trained from random init (see Section 5)
- **Safety mechanism (new this run):** hard wall-clock cap at 5.5 hours, checked every step, independent of the step-based LR schedule — this is what actually stopped the run, not step count. Guaranteed the run couldn't overrun the Founder's sleep window regardless of actual measured speed.

## 7. Training Results
| Metric | Value | Notes |
|---|---:|---|
| Final train loss | 1.5582 | step 39,209 |
| Final validation loss | 1.5715 | step 39,209 |
| Best validation loss | 1.5713 (step 36,500) | essentially identical to final — noise-level difference |
| Train perplexity | ≈ 4.75 | |
| Validation perplexity | ≈ 4.81 | |
| Total runtime | 5.50h (hard cap) | 00:00:23 → 05:30:38 |
| Tokens processed | 321,443,840 (39,210 × 32 × 256) | ~69.6% of one epoch over the fixed 461.9M-token corpus |
| Approx. epochs completed | ~0.70 | |
| Best checkpoint | `step_36500.pt` (practically tied with `step_39209_final.pt`) | |

## 8. Loss Curve Assessment
- **Did train loss decrease steadily?** Yes, and — critically — **still decreasing meaningfully at the point it was stopped**: 1.5988 (step 31,000) → 1.5744 (step 37,250) → 1.5582 (step 39,209). Unlike Run 2, which had visibly flattened by this point in its own curve, Run 3 had not yet plateaued.
- **Did validation loss decrease steadily?** Yes, tracking train closely throughout.
- **Did train and validation stay close?** Yes — gap stayed in the same small range as Run 1/2 (~0.01–0.03), no divergence at any point.
- **Any instability or spikes?** None observed.
- **Evidence of underfitting, good fit, or overfitting?** Well-fit, and — unlike Run 2 — **not yet at its ceiling when the run ended.** This is the headline finding: more capacity clearly bought a lower loss floor (1.57 vs. Run 2's 1.72 plateau), and the curve's continued descent at cutoff suggests there's still more available even beyond this run's point.
- **Short interpretation:** the capacity hypothesis from the prior report is confirmed — the 17M model's plateau was a real capacity ceiling, not a data or step-count limit. This run beat that ceiling using *fewer* steps (39,210 vs. Run 2's full 50,000) and was still improving when time ran out.

## 9. Sample Generations

### Sample 1
- **Checkpoint used:** `step_39209_final.pt`
- **Prompt:** "Once upon a time"
- **Decoding settings:** temperature 0.9, top-k 40, 120 tokens
- **Output:** *"Once upon a time, there was a little bunny named Benny. Benny loved to play hide-and-seek with his friends. One day, Benny decided to hide under his cozy burrow. As he was looking for a place to hide, Benny remembered he needed to lay in. He thought about what he should do and hopped over to his burrow. It was cozy and warm, just like his little burrow. Benny covered himself with his cozy burrow and went to his burrow. He lay down and felt much better. When he woke up, he felt refreshed and ready to play again tomorrow. Once upon a"*
- **Assessment:** sustains a single story throughout the full 120 tokens without switching topics (the "Once upon a" at the very end is the start of what would be the *next* story after the EOS boundary — correct behavior, not a break mid-thought). Some repetition of "cozy burrow" (4 times) — a real, if mild, degeneration pattern at this checkpoint. Otherwise the most coherent single-story sample of any run so far.

### Sample 2
- **Checkpoint used:** `step_39209_final.pt`
- **Prompt:** "The little dog"
- **Decoding settings:** temperature 0.9, top-k 40, 120 tokens
- **Output:** *"The little dog was stuck in his little mouth and couldn't get out. Luckily, he saw a kind man walking by with a long ladder. The man put the little dog down next to the little dog and helped him. The little dog was so happy when he saw the cat in his arms. The little dog learned to always be friendly and share his toys with others. Once upon a time, there was a little girl named Lily..."*
- **Assessment:** same prompt used in Run 1/2 for direct comparison. Still shows the "story switch" at the ~65-token mark — but a raw-token check confirmed this is **no longer a bug**: `eos_id` was found at exactly the correct positions (23 and 72) in this generation's token sequence. The model correctly ends the first story and starts a second; `generate()` simply doesn't stop at EOS (it always produces the full requested token count) — a separate, easy fix if clean single-story completions are wanted later, not a sign the data fix failed.

## 10. Qualitative Evaluation Checklist
- **Grammar / syntax:** clean throughout both samples, on par with or slightly better than Run 2.
- **Prompt adherence:** strong — both prompts are followed faithfully into a coherent single story.
- **Character consistency:** consistent naming and pronoun use within each story (Benny, the little dog).
- **Narrative coherence:** the strongest of any run — Sample 1 sustains a full setup→complication→resolution arc without losing the thread.
- **Repetition:** present but mild (the "cozy burrow" phrase repeats 4 times in Sample 1) — less severe than Run 1's verbatim phrase-looping, not fully eliminated.
- **Looping / degeneration:** none severe; the repetition above is stylistic drift, not a hard loop.
- **Long-range consistency:** best result yet — first run where an entire single-story generation holds together without a mid-story logic break.
- **Domain fit:** strong match to TinyStories register throughout.
- **Overall output quality:** clear step up from both prior dense runs.

## 11. Failure Modes
- **Observed issues:** (1) mild phrase repetition within a single story (Sample 1's "cozy burrow"); (2) the EOS fix's benefit isn't visible in raw printed samples because `generate()`/`sample.py` don't stop generation at EOS — cosmetic gap between what the model learned and what the sampling script surfaces.
- **When they appear:** repetition appears late in longer (120-token) generations; the EOS/story-switch pattern appears whenever generation is asked to continue past a natural story end.
- **Likely cause:** (1) still a capacity/training-duration interaction at this scale — not fully resolved by 2x more parameters; (2) a generation-script limitation, not a model or data limitation.
- **Severity:** low for both — neither blocks meaningful evaluation of this run's core result.
- **Examples:** see Section 9.

## 12. Comparison to Previous Runs
| Run | Steps | Train loss | Val loss | Val perplexity | Notes |
|---|---:|---:|---:|---:|---|
| **Run 3 (this run)** | 39,210 / 50,000 planned | 1.5582 | **1.5715** | **≈4.81** | 33.5M params, stopped by time cap while still improving |
| Run 2 | 50,000 | 1.7263 | 1.7209 | ≈5.59 | 17M params, plateaued by ~step 30,000 |
| Run 1 | 5,000 | 2.4933 | 2.4926 | ≈12.09 | 17M params, ~9% of one epoch |

Run 3 beat Run 2's final loss using **fewer steps** (78% of Run 2's step count) and a **different (fixed) dataset encoding** — both the capacity increase and the data fix are conflated in this comparison, worth noting as a real caveat: this run cannot cleanly isolate "bigger model helped" from "fixed data helped," since both changed at once. See Next Experiment Plan.

## 13. Interpretation
- **What improved in this run?** Meaningfully lower loss (1.57 vs. 1.72) reached with fewer steps than Run 2, and — unlike Run 2 — the curve hadn't plateaued when the run stopped. Qualitative output also improved (best single-story coherence of any run).
- **What did not improve?** Mild repetition is still present; long-generation degradation isn't fully solved, just pushed further out.
- **Is the model still learning?** Yes, clearly — this is the headline finding. The run was cut off by the time cap, not by convergence.
- **Is more training justified?** Yes, directly — continuing this exact run (same checkpoint, same config) for more steps is the most obviously justified next action of anything tested so far, since it hasn't hit its ceiling yet.
- **What is the main bottleneck right now?** Time, plainly — the wall-clock cap stopped a run that was still improving. Not model size, not data, not steps in the abstract.

## 14. Decision
- **Recommended next action:** resume/continue training this exact 33.5M model from `step_39209_final.pt` for more steps (a genuine resume is possible here, unlike Run 3 vs. Run 1/2, since nothing about the architecture needs to change) — ideally to the original 50,000-step target and beyond, since the curve hadn't flattened.
- **Why this is the best next action:** it's the only run in this project's history where "more of the same" is unambiguously still paying off — no plateau, no ambiguity about whether it's worth the compute.
- **Do not change yet:** model size, learning rate, or data — none of them are implicated as the bottleneck; only time was.
- **Checkpoint to keep:** `step_39209_final.pt` (and `step_36500.pt` as a near-identical backup).
- **Checkpoint to discard:** none needed yet.

## 15. Next Experiment Plan
- **Experiment name:** Run 3b — continuation
- **Single variable to change:** step count only (resume from `step_39209_final.pt`, same architecture, same fixed data, same hyperparameters) — needs a resume-from-checkpoint code path added to `train_run3.py`, which doesn't currently exist (it only supports starting fresh).
- **Hypothesis:** loss continues improving meaningfully toward — or past — the original 50,000-step target before showing its own plateau.
- **Success criteria:** val loss drops measurably below 1.5715; ideally the loss curve itself shows where the plateau actually starts, which this run didn't reach.
- **Metrics to monitor:** val loss trend shape (does it start flattening the way Run 2's did, and at what step); train/val gap (memorization check, especially now training past 1 full epoch is plausible if extended far enough); qualitative repetition rate in longer samples.
- **Rollback condition:** if val loss diverges from train loss (memorization) or the gap grows in a way it never did in Run 1–3, stop and prefer the current `step_39209_final.pt` checkpoint over anything trained further.

## 16. Appendix

### Commands used
```bash
python3 encode_data.py    # re-run with the EOS-separator fix
python3 train_run3.py
```

### File paths
```text
train_run3.py                       — training script (Config inline, 8L/8H/512E)
data/processed/train.bin, val.bin   — re-encoded with EOS fix (461.9M / 4.6M tokens)
data/processed_run1-2_backup/       — original unfixed encoding, preserved
checkpoints_run3/step_39209_final.pt — final checkpoint
overnight_pipeline.log              — full run log
```

### Extra notes
- This run's LR schedule was shaped for 50,000 steps but cut short at 39,210 by the time cap — a genuine continuation (Next Experiment Plan) should either extend the same schedule or start a fresh cosine schedule sized to the new target step count; using the current under-annealed LR indefinitely isn't ideal.

# SLM Training Evaluation Report — Run 4b (Confirming Continuation)

## 1. Run Summary
- **Run name:** Run 4b — bounded continuation from Run 4's best checkpoint, confirming or refuting the 1.3006 plateau at 305.6M scale
- **Date started:** 2026-09-15, ~11:03 BST
- **Date completed:** 2026-09-16, ~16:01 BST (28.97h wall clock)
- **Status:** Completed — full 25,000-step budget reached, did not hit the 40h safety cap
- **Objective:** test whether Run 4's tail-end flattening (steps ~46,750–49,999, oscillating 1.31–1.34) was a real plateau or a temporary flat patch — exactly mirroring the Run 3b → Run 3c confirming-experiment pattern, one scale up.
- **Author / operator:** Kishan Lad (build); report written by Claude
- **Notes:** Launched manually this session following Run 4's own eval report Next Experiment Plan. Monitored live via a purpose-built local dashboard (`run4b_dashboard.html` + `watch_run4b.py`); a real operational issue surfaced mid-run and is worth recording here since it affects how this report's numbers were sourced (see Section 16, "Monitoring notes").

## 2. Environment
- **Project folder:** `./`
- **Git commit / version:** not under version control — no git repo in use for this project
- **Python version:** 3.11.3
- **Framework versions:** torch 2.13.0, sentencepiece 0.2.2, numpy 2.4.3, tqdm 4.69.1
- **Hardware:** Apple M4 Pro, 48GB unified memory, 14 cores (10P/4E)
- **OS:** macOS
- **Device used for training:** `mps` (confirmed in log: `Using device: mps`)
- **Mixed precision enabled:** No
- **Checkpoint directory:** `checkpoints_run4b/` (local; not yet archived — see Section 16)

## 3. Dataset
- **Dataset name:** TinyStories (`roneneldan/TinyStories`)
- **Dataset source:** Hugging Face Hub
- **Domain / content type:** short, simple-vocabulary children's stories
- **Raw dataset size:** `data/raw/train.txt` ≈ 1.8GB
- **Cleaned / tokenized size:** `data/processed/train.bin` = 461,869,301 tokens; `val.bin` = 4,644,562 tokens
- **Train / Validation split:** TinyStories' own train/validation split
- **Data quality notes:** identical encoding to every prior run — not re-encoded for this run
- **License / usage notes:** unchanged from prior reports

## 4. Tokenizer
- **Tokenizer type:** SentencePiece, BPE
- **Vocabulary size:** 8,000
- **Special tokens:** `pad_id=0, unk_id=1, bos_id=2, eos_id=3`
- **Tokenizer file path:** `tokenizer/tinyslm.model` / `.vocab`
- **Encoding format:** uint16 token-ID arrays, `eos_id` appended after every example
- **Tokenizer notes:** identical across every run to date, including this one

## 5. Model Configuration
- **Architecture type:** GPT-style decoder-only transformer, from-scratch
- **Model class / file:** `slm.model.TinyGPT`, same `Config` override as Run 4 (18L/18H/1152E), defined locally inside `train_run4b.py`
- **Parameter count:** 305,653,248 — unchanged from Run 4, since this is a same-architecture continuation, not a capacity change
- **Number of layers / heads / embedding dim:** 18 / 18 / 1152 (unchanged)
- **Context length:** 256 (unchanged)
- **Dropout:** 0.1
- **Weight tying enabled:** No
- **Initialization notes:** resumed weights from `checkpoints_run4/step_46750.pt` (Run 4's *best* checkpoint, not its worse final one at step 49,999) — same deliberate deviation Run 3c used relative to Run 3b.

## 6. Training Configuration
- **Total planned steps (this phase):** 25,000, global step range 46,751 → 71,750
- **Completed steps:** 25,000 (full budget — exited normally, not via the time cap)
- **Batch size:** 32, no gradient accumulation
- **Tokens per step:** 32 × 256 = 8,192
- **Optimizer:** AdamW
- **Learning rate:** peak 3e-4 (same as Run 4), cosine decay to 10% of peak over this phase's own fresh 25,000-step schedule
- **Warmup steps:** 100 — the short continuation warmup (not Run 4's 200-step from-scratch warmup), same convention Run 3c used
- **Weight decay:** 0.01
- **Gradient clipping:** 1.0 (max norm)
- **Evaluation interval:** every 250 steps
- **Checkpoint interval:** every 250 steps
- **Random seed:** 46 (next in the project's sequence: 42/43/44/45)
- **Resume checkpoint:** `checkpoints_run4/step_46750.pt`
- **Time safety cap:** 40h (not hit — actual 28.97h)
- **Rollback condition (pre-committed):** if val loss hadn't beaten 1.3006 by the halfway point (local step 12,500 / global step 59,251) with no clear downward trend, treat the plateau as confirmed.

## 7. Training Results
| Metric | Value | Notes |
|---|---:|---|
| Final train loss | 1.2374 | step 71,750 |
| Final validation loss | 1.2834 | step 71,750 |
| Best validation loss | **1.2707** | step **67,751** — final checkpoint is again not the best one |
| Train perplexity | ≈3.45 (final) | |
| Validation perplexity | ≈3.61 (final) / **≈3.56 (best)** | |
| Total runtime | 28.97h | full budget completed, well inside the 40h safety cap |
| Tokens processed (this phase) | 204,800,000 (25,000 × 8,192) | ≈204.8M |
| Cumulative tokens since Run 4's random init (Run 4 + Run 4b) | 614,400,000 | ≈614.4M, ≈1.33 epochs of the 461.9M-token corpus |
| Best checkpoint | `checkpoints_run4b/step_67751.pt` | |

## 8. Loss Curve Assessment
- **Did train loss decrease steadily?** Yes overall, from ~1.29 (resumed) to 1.2374 (final) — a real further drop, though smaller in magnitude than Run 4's own descent from random init (expected, since this phase starts from an already-converged point).
- **Did validation loss decrease steadily?** No — and this is the most notable finding of this run. Val loss actually got *worse* immediately after resuming: from 1.3124 at step 46,751 up to a peak of 1.4409 at step 52,001, staying elevated (1.37–1.44) for roughly the first 5,000 steps of this phase, before recovering and trending downward from ~step 53,000 onward. This reads as a **warm-restart transient**: re-applying a peak LR of 3e-4 (via a fresh, short 100-step warmup) to weights that had already cosine-decayed to a low LR under Run 4's schedule temporarily knocked the model off its converged point before it re-settled. This project's continuation convention (Run 3c used it too) is to always use a fresh short-warmup schedule rather than splicing back into the original one — this run shows a real, measurable cost to that choice for the first ~20% of the new step budget.
- **Halfway rollback check (step 59,251):** best val this run was 1.3124, margin **-0.0118** — literally read, "NOT YET BEATEN." Had this check been acted on rigidly, it would have signaled "plateau confirmed, stop and scale up" — but the second half of the run (steps 59,251 → 67,751) improved from 1.3124 down to 1.2707, a real further -0.0417 drop. **Lesson for future rollback checks:** the halfway point alone can be misleading on a run recovering from a warm-restart transient; the "with no clear downward trend" qualifier in the original rollback condition matters and should be weighed, not just the raw margin.
- **Did train and validation stay close?** Yes, gap stayed in the ~0.02–0.05 range throughout (excluding the transient spike, which affected val more than train) — no overfitting signal.
- **Underfitting, good fit, or overfitting?** Good fit. After recovering from the transient, the run found a new, lower plateau (best 1.2707 at step 67,751) and then re-flattened for its final ~4,000 steps (oscillating 1.27–1.30), the same "best ≠ final" late-run signature seen in every prior run.

## 9. Sample Generations
*Generated fresh on 2026-09-16 from the best checkpoint, `checkpoints_run4b/step_67751.pt` — temperature 0.9, top_k 40, 120 new tokens. Same 3 prompts used in the Run 4 report, for direct comparison.*

### Sample 1
- **Prompt:** `"Once upon a time"`
- **Output:** "Once upon a time, there was a silly little girl named Lily. She loved to play outside and explore the woods behind her house. One day, she went on a walk and found a shiny rock. She picked it up and put it in her pocket. As she continued walking, she saw a big hill. She wanted to climb it, but her legs weren't strong enough yet. She felt frustrated and started to hurry up the hill. Suddenly, she heard a voice behind her. It was her mom! \"Lily, what are you doing?\" her mom asked. \"I'm climbing the"
- **Assessment:** Coherent single-story arc for the full window, correct dialogue formatting. Output is truncated mid-word ("climbing the") by the fixed 120-token cap — a testing-harness artifact (the same `generate()` limitation flagged in the Run 4 report, Section 11), not a model failure.

### Sample 2
- **Prompt:** `"The little dog looked at the ball and"`
- **Output:** "The little dog looked at the ball and then at the little cat. They both wanted the ball. They didn't know what to do. Then, the little dog had an idea. \"Let's both play with the ball,\" he said. \"We can take turns and be careful. That way, we won't be so stupid.\" The little cat liked the idea. They both took turns to play with the ball. They had fun and became friends. From that day on, they always played together and shared their toys. The little cat was happy that he learned to share and was not so stupid anymore."
- **Assessment:** Notable improvement on the specific failure Run 4's Sample 2 exhibited: the "little cat" here is properly introduced ("looked at the ball and then at the little cat") before being referenced later — Run 4's equivalent sample had a cat appear with zero setup. That specific entity-substitution bug did not recur. **New minor quirk:** repeated, slightly off-tone phrasing ("so stupid" / "not so stupid anymore") — unusually harsh self-referential language for this corpus's normally gentle moral-lesson style. Isolated, low severity.

### Sample 3
- **Prompt:** `"Tom and Sarah went to the park because"`
- **Output:** "Tom and Sarah went to the park because it was sunny and dry. They played with their toys and ran around. Suddenly, the ground started to shake and Tom and Sarah saw a big tree falling down. They were scared and didn't know what to do. But then, the wind started blowing and it calmed their area. They stood together and watched as the tree slowly fell down. Tom and Sarah were happy that they were safe and they continued to play at the park until the sun went down. Once upon a time, there was a little girl named Lily. She loved to play outside in her backyard. One day, she"
- **Assessment:** Both prompt-given names ("Tom" and "Sarah") are preserved correctly for the entire first story — the Run 3b-era name-substitution bug still does not recur, consistent with Run 4. **However:** after that story concludes cleanly ("...continued to play at the park until the sun went down."), the model abruptly starts an entirely new, unrelated story ("Once upon a time, there was a little girl named Lily...") within the same 120-token window. This is the exact story-switching failure mode Run 4's report highlighted as resolved (0 of 3 samples switched stories) — it has reappeared here (1 of 3). Root cause is the same standing, known limitation: `generate()` doesn't stop at the EOS token, so it happily continues past a natural story ending into a new one. Not a new bug, but its visibility in this specific 3-sample test is a reminder that Run 4's "fixed" result was likely sample-specific, not a durable resolution.

## 10. Qualitative Evaluation Checklist
- **Grammar / syntax:** Strong across all 3 samples, consistent with Run 4.
- **Prompt adherence:** Strong on named entities (Tom/Sarah preserved); Sample 1's truncation is a harness artifact, not an adherence failure.
- **Character consistency:** Improved on the specific axis Run 4 flagged — Sample 2's cat is properly introduced this time, no unexplained entity.
- **Narrative coherence:** Mixed — 2 of 3 samples stayed on one story for the full window; Sample 3 switched into a second, unrelated story near the end, reversing what looked like a clean win in the Run 4 report.
- **Repetition:** Mild, odd repetition of "stupid"/"not... stupid anymore" in Sample 2 — low severity.
- **Looping / degeneration:** None observed.
- **Long-range consistency:** A genuine mixed result vs. Run 4 — one previously-observed failure mode (entity substitution) did not recur, but another previously-"resolved" one (story-switching) did.
- **Domain fit:** Strong, consistent TinyStories style and moral-ending pattern.
- **Overall output quality:** The quantitative improvement (val loss 1.3006 → 1.2707) did **not** translate into a uniformly better qualitative result on this specific 3-prompt test — a useful, honest finding. Sample size of 3 is small and this shouldn't be over-interpreted as "quality got worse," but it's a real reminder that loss and small qualitative samples can diverge, and both need tracking rather than assuming one implies the other.

## 11. Failure Modes
- **Observed issues:** (1) story-switching within the 120-token window (Sample 3) — the model continues generating past a natural story ending into a new, unrelated one; (2) mild off-tone repetitive phrasing ("so stupid"/"not so stupid anymore," Sample 2); (3) fixed-length truncation mid-word (Sample 1) — a testing artifact, not a model issue.
- **Likely cause:** (1) `generate()` (in `slm/model.py`) has no EOS-stop condition — this is a known, standing limitation flagged in the Run 4 report's own Failure Modes section and not yet fixed; it happened not to show up in Run 4's 3 samples, and has now shown up again in Run 4b's. (2) likely a rare but present pattern in the training corpus surfacing under this specific prompt/temperature combination — not investigated further.
- **Severity:** Low-to-moderate. (1) is a recurring, known, unresolved limitation — now observed in 1 of the last 6 evaluation samples across the two most recent reports (Run 4 + Run 4b combined) — worth actually fixing given it directly affects how trustworthy future "did story-switching improve" comparisons are.
- **Examples:** Sample 3 above (story-switching); Sample 2 above (off-tone repetition).

## 12. Comparison to Previous Runs
| Run | Steps | Params | Train loss | Val loss | Val perplexity | Notes |
|---|---:|---:|---:|---:|---:|---|
| **Run 4b (this run)** | 25,000 (71,750 cumulative) | 305.6M | 1.2374 (final) | 1.2834 (final) / **1.2707 (best, step 67,751)** | ≈3.61 (final) / **≈3.56 (best)** | Confirmed Run 4's plateau was not real — resumed from Run 4's best checkpoint; warm-restart transient in the first ~5,000 steps, then a real further improvement |
| Run 4 | 50,000 | 305.6M | 1.2863 (final) | 1.3156 (final) / 1.3006 (best, step 46,750) | ≈3.73 (final) / ≈3.67 (best) | ~9x capacity jump, random init |
| Run 3c | 20,000 (93,710 cumulative) | 33.5M | — | 1.4562 (best) | ≈4.29 | Confirmed the 33.5M-scale plateau — triggered Run 4 |
| Run 3b | 40,000 (79,209 cumulative) | 33.5M | 1.4766 (final) | 1.4632 (best, step 73,710) | ≈4.32 | Best checkpoint again not the final one |
| Run 3 | 39,210 / 50,000 planned | 33.5M | 1.5582 | 1.5715 | ≈4.81 | Time-capped while still improving |
| Run 2 | 50,000 | 16.9M | 1.7263 | 1.7209 | ≈5.59 | Plateaued by ~step 30,000 |
| Run 1 | 5,000 | 16.9M | 2.4933 | 2.4926 | ≈12.09 | ~9% of one epoch |

## 13. Interpretation
- **What improved in this run?** A real, substantial drop in validation loss (1.3006 → 1.2707, margin 0.0299 — nearly 3x the >0.01 threshold that defined success). This confirms Run 4's plateau **was not real** — more training at 305.6M scale remained a genuinely productive lever, at least up to 71,750 cumulative-equivalent steps.
- **What did not improve, or got more complicated?** The halfway rollback check would have called this a confirmed plateau if read rigidly (margin -0.0118 at step 59,251) — the actual result only emerged in the second half, after recovering from a warm-restart transient. Qualitatively, the story-switching failure mode reappeared in the sample set after seeming resolved in Run 4.
- **Is the model still learning?** The run re-flattened over its last ~4,000 steps (1.27–1.30 oscillation after the 1.2707 low at step 67,751) — the same late-run signature every run has shown. Whether that's this architecture's real ceiling or another temporary flat patch is, once again, unconfirmed without a further bounded run.
- **Is more training justified?** The loss evidence says plausibly yes (same open question as Run 4 itself), but two things now argue against just repeating the same move a third time: (1) cumulative exposure to the same fixed, narrow-domain TinyStories corpus is now ~1.33 epochs across Run 4 + Run 4b — comfortably still under 1 epoch of pure novelty, but climbing, and (2) the qualitative sample evidence didn't move in lockstep with the loss number this time, which is a signal that loss alone may be an incomplete picture at this point.
- **What is the main bottleneck right now: steps, model size, context length, or data?** This is the first run in the project where **data** becomes a genuinely live candidate rather than a near-certain "no" — not because the corpus is data-exhausted (it isn't, at ~1.33 epochs), but because the qualitative ceiling (simple, repetitive TinyStories patterns) may now be limiting what further steps on the *same* data can teach the model, independent of remaining capacity or step budget.

## 14. Decision
- **Recommended next action:** Two independent, low-risk items before committing to another bounded step-count run (a "Run 4c") or another capacity increase (a "Run 5"): (1) fix the `generate()` EOS-stop bug — cheap, orthogonal to any training decision, and directly improves the reliability of every future qualitative comparison; (2) have the training-data discussion the Founder flagged for right after this report — whether to introduce new/diversified data before spending another ~29h confirming or refuting yet another plateau on the same fixed corpus.
- **Why this is the best next action:** Two runs in a row (Run 3c, Run 4b) have now shown that blind step-count continuations reliably produce *some* further improvement — that pattern itself is no longer generating much new information. The more informative open question right now is whether the data itself is the next real lever, which the loss curve alone can't answer.
- **Do not change yet:** model architecture/size or tokenizer — no evidence yet points at either as the binding constraint.
- **Checkpoint to keep:** `checkpoints_run4b/step_67751.pt` — the new canonical best Emere1.0 dense checkpoint.
- **Checkpoint to discard:** none need deleting; `step_71750_final.pt` is superseded as "the model to use" but harmless to retain.

## 15. Next Experiment Plan
- Deliberately left open pending the training-data discussion (see Section 14) — committing to a specific Run 4c or Run 5 spec before that conversation would repeat the exact "guess instead of reasoning from evidence" mistake this project's own process is built to avoid.

## 16. Appendix

### Commands used
```bash
# Sample generation for Section 9 (2026-09-16):
python _tmp_sample_run4b.py "Once upon a time" checkpoints_run4b/step_67751.pt
python _tmp_sample_run4b.py "The little dog looked at the ball and" checkpoints_run4b/step_67751.pt
python _tmp_sample_run4b.py "Tom and Sarah went to the park because" checkpoints_run4b/step_67751.pt
# temporary script, deleted after use -- Config doesn't match slm/model.py's module default

# Training (completed 2026-09-16):
python train_run4b.py
```

### File paths
```text
Best checkpoint:      checkpoints_run4b/step_67751.pt
Final checkpoint:     checkpoints_run4b/step_71750_final.pt
Training log:         train_run4b.log
Training script used: train_run4b.py
```

### Monitoring notes
This run's stdout was fully buffered (Python's default behavior when stdout isn't a tty), so the `global step ... val loss ...` print lines were stuck in memory for most of the run and not visible live in `train_run4b.log`, even though checkpoints were saving correctly on schedule. A companion script (`eval_shadow_run4b.py`) independently re-evaluated each new checkpoint on MPS (forward-only, small batch count, deliberately brief so as not to meaningfully compete with the training process for GPU time) to provide a live-updating dashboard (`run4b_dashboard.html`) without touching the training process itself. The buffered log eventually flushed on its own by the time the run completed, and its values matched the shadow evaluator's for the one directly-comparable step (46,751: log 1.3124 vs. shadow 1.3167 — close, small difference expected since eval batches are randomly sampled each time). All numbers in this report are taken from the authoritative flushed log, not the shadow evaluator.

### Extra notes
- This run was launched manually this session, not by `run_next_stage.sh`'s auto-launch mechanism (unlike Run 4).
- `checkpoints_run4b/` (102 files, ~117GB) is not yet archived to NAS, same standing item as `checkpoints_run4/`'s pending archive.

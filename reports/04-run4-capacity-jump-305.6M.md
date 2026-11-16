# SLM Training Evaluation Report — Run 4 (305.6M Capacity Jump)

## 1. Run Summary
- **Run name:** Run 4 — deliberate ~9x capacity increase over Run 3/3b/3c
- **Date started:** 2026-09-11, 20:48 BST
- **Date completed:** 2026-09-14, ~04:44 BST (55.92h wall clock)
- **Status:** Completed — full 50,000-step budget reached, did not hit the 90h safety cap
- **Objective:** test whether a much larger model breaks past the plateau confirmed by Run 3c (best val loss 1.4562, only 0.007 better than Run 3b's 1.4632 — short of the >0.01 bar that would justify continuing at 33.5M scale). Same reasoning that motivated Run 2 → Run 3.
- **Author / operator:** Kishan Lad (build); report written by Claude
- **Notes:** Auto-launched by `run_next_stage.sh` the instant Run 3c's result confirmed the plateau — the Founder had pre-committed to this exact architecture and step budget before Run 3c's result was known, conditional on the plateau being real.

## 2. Environment
- **Project folder:** `./`
- **Git commit / version:** not under version control — no git repo in use for this project
- **Python version:** 3.11.3
- **Framework versions:** torch 2.13.0, sentencepiece 0.2.2, numpy 2.4.6, tqdm 4.69.1
- **Hardware:** Apple M4 Pro, 48GB unified memory, 14 cores (10P/4E)
- **OS:** macOS
- **Device used for training:** `mps` (confirmed in log: `Using device: mps`)
- **Mixed precision enabled:** No
- **Checkpoint directory:** `checkpoints_run4/` (local during training; archived to NAS post-completion, see Section 16)

## 3. Dataset
- **Dataset name:** TinyStories (`roneneldan/TinyStories`)
- **Dataset source:** Hugging Face Hub
- **Domain / content type:** short, simple-vocabulary children's stories
- **Raw dataset size:** `data/raw/train.txt` ≈ 1.8GB
- **Cleaned / tokenized size:** `data/processed/train.bin` = 461,869,301 tokens; `val.bin` = 4,644,562 tokens
- **Train / Validation split:** TinyStories' own train/validation split
- **Estimated total tokens:** ~466.5M (train + val)
- **Data quality notes:** identical encoding to Run 3/3b/3c — EOS-separator fix already applied, not re-encoded for this run
- **License / usage notes:** unchanged from prior reports

## 4. Tokenizer
- **Tokenizer type:** SentencePiece, BPE
- **Tokenizer training corpus:** TinyStories `train.txt` (same tokenizer as every prior run — not retrained)
- **Vocabulary size:** 8,000
- **Special tokens:** `pad_id=0, unk_id=1, bos_id=2, eos_id=3`
- **Tokenizer file path:** `tokenizer/tinyslm.model` / `.vocab`
- **Encoding format:** uint16 token-ID arrays, `eos_id` appended after every example
- **Tokenizer notes:** identical across every run to date, including this one

## 5. Model Configuration
- **Architecture type:** GPT-style decoder-only transformer, from-scratch (no pretrained weights, no `transformers` library)
- **Model class / file:** `slm.model.TinyGPT`, instantiated with a `Config` class defined locally inside `train_run4.py` — a deliberate override of `slm/model.py`'s module-level default (6L/6H/384E)
- **Parameter count:** **305,653,248** — confirmed by direct calculation and matching the pre-launch estimate
- **Number of layers:** 18 (was 8 in Run 3/3b/3c)
- **Number of attention heads:** 18 (was 8)
- **Embedding dimension:** 1152 (was 512) — `head_dim = 1152/18 = 64`, preserving the same head_dim every prior run has used (384/6, 512/8)
- **Context length:** 256 (unchanged)
- **Dropout:** 0.1
- **Activation function:** GELU (in the MLP)
- **Positional encoding:** learned absolute positional embeddings
- **Weight tying enabled:** No
- **Initialization notes:** random init — every architectural dimension changed from Run 3c, so none of its weights were shape-compatible; no upcycling was possible for this run

## 6. Training Configuration
- **Total planned steps:** 50,000
- **Completed steps:** 50,000 (full budget — the process exited normally via `max_steps`, not the time cap)
- **Batch size:** 32
- **Gradient accumulation:** none
- **Effective batch size:** 32
- **Tokens per step:** 32 × 256 = 8,192
- **Optimizer:** AdamW
- **Learning rate:** peak 3e-4, same as every prior dense run
- **Learning rate schedule:** cosine decay to 10% of peak over the full 50,000-step budget
- **Warmup steps:** 200 (matches Run 3's from-scratch warmup, since this is also training from random init — not Run 3b/3c's 100-step continuation warmup)
- **Weight decay:** 0.01
- **Gradient clipping:** 1.0 (max norm)
- **Evaluation interval:** every 250 steps
- **Checkpoint interval:** every 250 steps
- **Random seed:** 45 (next in the project's sequence: 42/43/44)
- **Resume from checkpoint:** No — random init

## 7. Training Results
| Metric | Value | Notes |
|---|---:|---|
| Final train loss | 1.2863 | step 49,999 |
| Final validation loss | 1.3156 | step 49,999 |
| Best validation loss | **1.3006** | step **46,750** — final checkpoint is again not the best one, consistent with Run 3b's pattern |
| Train perplexity | ≈3.62 (final) | |
| Validation perplexity | ≈3.73 (final) / **≈3.67 (best)** | |
| Total runtime | 55.92h | full budget completed, well inside the 90h safety cap |
| Tokens processed | 409,600,000 (50,000 × 8,192) | ≈409.6M |
| Approx. epochs completed | ≈0.89 | vs. the 461.9M-token train corpus — still under one full epoch |
| Best checkpoint | `checkpoints_run4/step_46750.pt` | |

## 8. Loss Curve Assessment
- **Did train loss decrease steadily?** Yes, from 9.14 (random init) down to ~1.27–1.29 by the end — no instability at any point in the log.
- **Did validation loss decrease steadily?** Yes overall, reaching its lowest point (1.3006) at step 46,750, then oscillating in the 1.31–1.34 band for the remaining ~3,250 steps rather than continuing to fall — the same late-run flattening signature seen in Run 3b, just arriving proportionally later in this run's budget (~93.5% of the way through, vs. Run 3b's ~86%).
- **Did train and validation stay close?** Yes, consistently within ~0.02–0.04 — no overfitting signal.
- **Any instability or spikes?** None observed in the eval-log record.
- **Underfitting, good fit, or overfitting?** Good fit, again trending toward a plateau by the very end — but importantly, this plateau sits meaningfully lower (1.30) than Run 3c's plateau (1.4562), confirming the capacity increase itself produced real headroom, not just a shifted noise floor.
- **Short interpretation:** the same "did the model still have room to improve, or was capacity now sufficient" question that was asked before Run 4 can be asked again here — the late flattening suggests this scale may itself approach a plateau with enough further steps, though it hasn't been confirmed the way Run 3c confirmed the 33.5M plateau.

## 9. Sample Generations
*Generated fresh on 2026-09-14 from the best checkpoint, `checkpoints_run4/step_46750.pt` — temperature 0.9, top_k 40, 120 new tokens.*

### Sample 1
- **Checkpoint used:** `step_46750.pt`
- **Prompt:** `"Once upon a time"`
- **Decoding settings:** temperature 0.9, top_k 40, max_new_tokens 120
- **Output:** "Once upon a time, there was a boy named Timmy. Timmy had a green wallet that he loved very much. One day, Timmy went to the park to play with his friends. He put his green wallet on a bench while he went to play. When he came back, his wallet was gone! Timmy was very sad. Timmy said, \"Oh no! My green wallet!\" His friend, Sally, said, \"I'm sorry, Timmy. I think it disappeared forever.\" Timmy didn't know what \"gain\" meant, but he knew it was bad. But he was happy to have"
- **Assessment:** Strong — a full, coherent single-story arc for the entire 120-token window (no story-switching, unlike every prior run's Sample 1). Correct quoted dialogue formatting with attribution ("Timmy said,", "Sally, said,"). One genuine oddity: "Timmy didn't know what 'gain' meant" — the word "gain" appears with no prior setup, likely a stray token the model associated with the loss/moral-lesson pattern common in TinyStories endings. Minor, not a coherence failure.

### Sample 2
- **Checkpoint used:** `step_46750.pt`
- **Prompt:** `"The little dog looked at the ball and"`
- **Decoding settings:** temperature 0.9, top_k 40, max_new_tokens 120
- **Output:** "The little dog looked at the ball and then at the big dog. The big dog wanted the ball too. The little dog and the big dog started to fight. The little dog pulled the big cat's tail. The big dog pulled his paw back. The little dog pulled his tail back. They both pulled the ball very hard. The big dog and the little dog pulled so hard that they broke the big dog's bone. They both looked at each other and felt sad. The big dog did not say sorry and walked away. The little dog and the big dog learned that fighting is not nice, and they wished"
- **Assessment:** Mostly coherent single story, correct moral-lesson closing (a consistent TinyStories pattern this model has clearly learned well). One clear entity-tracking error: "the big cat's tail" appears mid-scene where only "the little dog" and "the big dog" have been established — no cat was introduced. A real, specific failure mode worth naming: the model can lose track of which entities are actually in a scene during a fast-paced action sequence.
- **Failure mode identified:** entity substitution during multi-entity action scenes (a dog's tail momentarily becomes "the big cat's tail" with no cat present).

### Sample 3
- **Checkpoint used:** `step_46750.pt`
- **Prompt:** `"Tom and Sarah went to the park because"`
- **Decoding settings:** temperature 0.9, top_k 40, max_new_tokens 120
- **Output:** "Tom and Sarah went to the park because they were very hungry. Sarah said to her Mom, \"I am so hungry, can we get something to eat?\" Her Mom smiled and said, \"Let's go get some ice-cream\". At the ice-cream shop, the shopkeeper gave them each an ice-cream. The ice-cream was thick, creamy and yummy. Sarah and Tom ate their ice-cream and were very grateful. \"Let's always remember this day,\" said Sarah's Mom. They waved goodbye to the shopkeeper, and then went home. Sarah and Tom smiled to"
- **Assessment:** The strongest sample of the three, and a direct, measurable improvement over Run 3b's equivalent test: **both prompt-given names ("Tom" and "Sarah") are preserved and used consistently for the entire 120-token output** — Run 3b's report specifically flagged this exact prompt substituting the default name "Lily" within the first sentence; Run 4 does not repeat that failure. Full coherent arc, correct quoted dialogue with attribution, consistent character relationships ("her Mom," "Sarah's Mom") throughout.

## 10. Qualitative Evaluation Checklist
- **Grammar / syntax:** Strong across all 3 samples, including correctly punctuated multi-speaker dialogue — a step up in surface polish from Run 3b's samples.
- **Prompt adherence:** Strong — the specific character-name failure documented in the Run 3b report (Section 9, Sample 3 of that report) did **not** recur here on the same prompt. Direct, measurable improvement.
- **Character consistency:** Good within each sample; Sample 2 shows one clear lapse (see Failure Modes).
- **Narrative coherence:** Notably improved — all 3 samples stayed on a single story for the full 120-token window, unlike Run 3b where 2 of 3 samples ran on into an unrelated second story.
- **Repetition:** None observed.
- **Looping / degeneration:** None observed.
- **Long-range consistency:** Improved over Run 3b, though not perfect — Sample 2's stray "big cat" shows entity tracking is still imperfect under multi-entity action scenes specifically.
- **Domain fit:** Strong, consistent TinyStories style and moral-ending pattern.
- **Overall output quality:** A clear, visible qualitative improvement over Run 3b's samples, not just a lower loss number — longer coherent spans, better dialogue formatting, and the specific previously-documented name-substitution failure did not reproduce.

## 11. Failure Modes
- **Observed issues:** (1) one instance of entity substitution mid-scene ("the big cat's tail" with no cat present); (2) one instance of an unexplained word choice with no narrative setup ("gain"); (3) the underlying `generate()` limitation (doesn't stop at EOS) wasn't tested at longer generation lengths this round since all 3 samples stayed on-topic for the full 120 tokens — worth retesting at a longer `max_new_tokens` to see if story-switching still occurs beyond this window.
- **When they appear:** (1) during scenes with 2+ named/described entities acting simultaneously; (2) near narrative transitions where the model may be drawing on a "lesson" pattern from training data.
- **Likely cause:** (1) at 256 tokens of context and this scale, tracking exactly which entities are "in scene" during fast action sequences is still an imperfect learned behavior, not solved by this capacity increase alone; (2) likely a case of the model pattern-matching to common TinyStories moral-lesson vocabulary without full contextual grounding.
- **Severity:** Low — both are minor, isolated errors in otherwise coherent output, a genuine improvement over the failure rate/severity seen in Run 3b's samples.
- **Examples:** Sample 2 above (entity substitution); Sample 1 above (ungrounded word choice).

## 12. Comparison to Previous Runs
| Run | Steps | Params | Train loss | Val loss | Val perplexity | Notes |
|---|---:|---:|---:|---:|---:|---|
| **Run 4 (this run)** | 50,000 | 305.6M | 1.2863 (final) / — (best-ckpt train n/a) | 1.3156 (final) / **1.3006 (best, step 46,750)** | ≈3.73 (final) / **≈3.67 (best)** | ~9x capacity jump, random init, confirmed real improvement over the 33.5M plateau |
| Run 3c | 20,000 (93,710 cumulative) | 33.5M | — | 1.4562 (best) | ≈4.29 | Confirmed the 33.5M-scale plateau — this result triggered Run 4 |
| Run 3b | 40,000 (79,209 cumulative) | 33.5M | 1.4766 (final) | 1.4632 (best, step 73,710) | ≈4.32 | Best checkpoint again not the final one |
| Run 3 | 39,210 / 50,000 planned | 33.5M | 1.5582 | 1.5715 | ≈4.81 | Time-capped while still improving |
| Run 2 | 50,000 | 16.9M | 1.7263 | 1.7209 | ≈5.59 | Plateaued by ~step 30,000 |
| Run 1 | 5,000 | 16.9M | 2.4933 | 2.4926 | ≈12.09 | ~9% of one epoch |

## 13. Interpretation
- **What improved in this run?** A real, substantial drop in validation loss (1.4562 → 1.3006, a much larger margin than the 0.01 threshold that would have justified staying at 33.5M) and a visible qualitative jump in sample quality — longer coherent spans, correct dialogue punctuation, and resolution of the specific name-substitution failure documented in the Run 3b report.
- **What did not improve?** The entity-tracking failure mode (Sample 2's stray "cat") is new/newly-observed at this scale, though minor. The last ~3,250 steps show the same late-run flattening pattern as Run 3b, just arriving later in the schedule.
- **Is the model still learning?** Likely yes for at least part of the remaining budget, though the tail-end flattening (steps ~46,750–49,999) raises the same question Run 3c was built to answer for the previous scale — this hasn't been tested yet at this scale.
- **Is more training justified?** Plausibly, but unconfirmed — the honest, disciplined answer (per this project's own standing pattern) is to run the same kind of bounded confirming experiment used for Run 3c before assuming either "train more" or "scale up again" is correct.
- **What is the main bottleneck right now: steps, model size, context length, or data?** Unclear without a confirming run — this result doesn't yet distinguish between "this scale needs more steps to find its real plateau" and "this scale's plateau is already close to being reached." Data volume is very unlikely to be the constraint: only ≈0.89 epochs of the corpus have been seen in this run alone.

## 14. Decision
- **Recommended next action:** Run a bounded continuation from the best checkpoint (`step_46750.pt`), mirroring exactly the Run 3b → Run 3c pattern at this new scale, to determine whether the tail-end flattening here is a real plateau or a temporary flat patch — before deciding whether the next move is more training or another capacity increase.
- **Why this is the best next action:** This exact process (confirm-before-scaling) already worked once in this project (Run 3c's result correctly triggered Run 4) — reusing a validated decision procedure rather than guessing.
- **Do not change yet:** model architecture/size, tokenizer, or dataset.
- **Checkpoint to keep:** `checkpoints_run4/step_46750.pt` — the new canonical best Emere1.0 dense checkpoint.
- **Checkpoint to discard:** none need deleting; `step_49999_final.pt` is superseded as "the model to use" but harmless to retain.

## 15. Next Experiment Plan
- **Experiment name:** Run 4b — bounded continuation from `step_46750`, confirming the plateau at 305.6M scale
- **Single variable to change:** step count only, resuming from the best checkpoint (46,750), not the run's final one (49,999) — same deliberate deviation used for Run 3c.
- **Hypothesis:** a further bounded number of steps will either (a) push validation loss measurably below 1.3006 (more than a noise-level margin) — supporting "keep training at 305.6M" — or (b) re-plateau/oscillate near 1.30–1.34 — supporting a further capacity increase next.
- **Success criteria:** a new best validation loss more than 0.01 below 1.3006 within the new step budget (same margin used for the Run 3c decision).
- **Metrics to monitor:** validation loss every 250 steps; also worth tracking whether Sample 2's entity-substitution failure mode persists, decreases, or worsens with further training at this scale.
- **Rollback condition:** if validation loss has not beaten 1.3006 by the halfway point of the new step budget with no clear downward trend, treat the plateau as confirmed.

## 16. Appendix

### Commands used
```bash
# Sample generation for Section 9 (2026-09-14):
python _tmp_sample_run4.py   # temporary script, deleted after use — Config
                              # doesn't match slm/model.py's module default

# Training (completed 2026-09-14):
python train_run4.py
```

### File paths
```text
Best checkpoint:      checkpoints_run4/step_46750.pt
Final checkpoint:     checkpoints_run4/step_49999_final.pt
Training log:         train_run4.log
Training script used: train_run4.py
```

### Extra notes
- This run was auto-launched by `run_next_stage.sh` based on Run 3c's confirmed plateau — see emere1-slm memory for the full conditional-launch mechanism.
- 232GB of checkpoints (202 files) are pending archive to the NAS following the same rsync+verify+remove pattern used for Runs 2/3/3b/3c/MoE.

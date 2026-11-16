# SLM Training Evaluation Report — Run 3b (Extended Continuation)

## 1. Run Summary
- **Run name:** Run 3b — extended continuation of Run 3
- **Date started:** 2026-07-30, 07:38 BST
- **Date completed:** 2026-07-30, ~13:20 BST (start + 5.67h wall clock)
- **Status:** Completed — full 40,000-step local budget reached, did not hit the 6h time cap
- **Objective:** Continue Run 3 past the point where its 5.5h wall-clock cap stopped it (Run 3 was still improving, not plateaued, per that report's own Next Experiment Plan: "single variable to change: step count only")
- **Author / operator:** Kishan Lad (build); report written by Claude
- **Notes:** This report was never written at the time (a standing process gap — see feedback_slm_eval_report_process). It is being written retroactively on 2026-09-11, ~6 weeks after the run finished, entirely from preserved artifacts: `train_run3b.log` (full per-eval loss log) and the `checkpoints_run3b/` directory. No numbers here are estimated or recalled from memory — every figure below was re-derived directly from those files. The only genuinely new content is Section 9 (sample generations), which was generated fresh today from the preserved best checkpoint.

## 2. Environment
- **Project folder:** `./`
- **Git commit / version:** not under version control — no git repo in use for this project
- **Python version:** 3.11.3
- **Framework versions:** torch 2.13.0, sentencepiece 0.2.2, numpy 2.4.6, tqdm 4.69.1
- **Hardware:** Apple M4 Pro, 48GB unified memory, 14 cores (10P/4E)
- **OS:** macOS
- **Device used for training:** `mps` (confirmed in log: `Using device: mps`)
- **Mixed precision enabled:** No
- **Checkpoint directory:** `checkpoints_run3b/`

## 3. Dataset
- **Dataset name:** TinyStories (`roneneldan/TinyStories`)
- **Dataset source:** Hugging Face Hub
- **Domain / content type:** short, simple-vocabulary children's stories
- **Raw dataset size:** `data/raw/train.txt` ≈ 1.8GB
- **Cleaned / tokenized size:** `data/processed/train.bin` = 923,738,602 bytes → **461,869,301 tokens** (uint16); `val.bin` = 9,289,124 bytes → **4,644,562 tokens**
- **Train / Validation split:** TinyStories' own train/validation split (no test split used)
- **Estimated total tokens:** ~466.5M (train + val)
- **Data quality notes:** unchanged from Run 3 — this run reuses Run 3's already EOS-separator-fixed encoding (see emere1-slm, 2026-07-29 fix); no re-encoding was done for Run 3b
- **License / usage notes:** not independently re-verified for this report; carried forward from prior reports

## 4. Tokenizer
- **Tokenizer type:** SentencePiece, BPE
- **Tokenizer training corpus:** TinyStories `train.txt` (same tokenizer as every prior run — not retrained)
- **Vocabulary size:** 8,000
- **Special tokens:** `pad_id=0, unk_id=1, bos_id=2, eos_id=3`
- **Tokenizer file path:** `tokenizer/tinyslm.model` / `.vocab`
- **Encoding format:** uint16 token-ID arrays, `eos_id` appended after every example
- **Tokenizer notes:** identical across Run 1 through Run 3b

## 5. Model Configuration
- **Architecture type:** GPT-style decoder-only transformer, from-scratch (no pretrained weights, no `transformers` library)
- **Model class / file:** `slm.model.TinyGPT`, instantiated with a `Config` class defined locally inside `train_run3b.py` — **not** `slm/model.py`'s module-level default `Config` (that default is 6L/6H/384E; this run overrides to 8L/8H/512E)
- **Parameter count:** **33,543,168** (confirmed directly in log: `Model parameters: 33,543,168`) — identical to Run 3, since this is a direct weight continuation of it
- **Number of layers:** 8
- **Number of attention heads:** 8
- **Embedding dimension:** 512
- **Context length:** 256
- **Dropout:** 0.1
- **Activation function:** GELU (in the MLP)
- **Positional encoding:** learned absolute positional embeddings
- **Weight tying enabled:** No — `lm_head` is a separate `nn.Linear`, not tied to the token embedding
- **Initialization notes:** **not** random init — resumed from Run 3's checkpoint `checkpoints_run3/step_39209_final.pt` (log: `Resumed weights from checkpoints_run3/step_39209_final.pt (was step 39209)`)

## 6. Training Configuration
- **Total planned steps:** 40,000 local steps (new budget for this continuation phase); cumulative global step target ≈79,210
- **Completed steps:** 40,000 local steps — full budget, **not** cut short by the time cap (finished in 5.67h vs. a 6h cap)
- **Batch size:** 32
- **Gradient accumulation:** none
- **Effective batch size:** 32
- **Tokens per step:** 32 × 256 = 8,192
- **Optimizer:** AdamW
- **Learning rate:** peak 3e-4 — same peak as Run 3, unchanged per that report's explicit instruction ("single variable to change: step count only")
- **Learning rate schedule:** cosine decay to 10% of peak; a **fresh, short-warmup schedule** for this continuation phase, not a splice back into Run 3's original 50k-step schedule
- **Warmup steps:** 100 (local)
- **Weight decay:** 0.01
- **Gradient clipping:** 1.0 (max norm)
- **Evaluation interval:** every 250 local steps
- **Checkpoint interval:** every 250 local steps
- **Random seed:** 43 (deliberately different from Run 1/2/3's seed 42 — affects only this phase's batch sampling order)
- **Resume from checkpoint:** Yes — `checkpoints_run3/step_39209_final.pt`

## 7. Training Results
| Metric | Value | Notes |
|---|---:|---|
| Final train loss | 1.4766 | global step 79,209 |
| Final validation loss | 1.4816 | global step 79,209 |
| Best validation loss | **1.4632** | global step **73,710** — the final checkpoint is *not* the best one |
| Train perplexity | ≈4.38 (final) / ≈4.32 (best-checkpoint) | |
| Validation perplexity | ≈4.40 (final) / **≈4.32 (best)** | |
| Total runtime | 5.67h | full 40,000-step budget completed, did not hit the 6h cap |
| Tokens processed (this run) | 327,680,000 (40,000 × 8,192) | ≈327.7M |
| Cumulative tokens since random init (Run 3 + Run 3b) | 648,888,320 | ≈648.9M |
| Approx. epochs completed | ≈0.71 (this run alone) / **≈1.40 cumulative** | vs. the 461.9M-token train corpus |
| Best checkpoint | `checkpoints_run3b/step_73710.pt` | |

## 8. Loss Curve Assessment
- **Did train loss decrease steadily?** Yes overall (1.5604 → 1.4766), but noisy step-to-step, as expected for random-batch SGD at this batch size.
- **Did validation loss decrease steadily?** Mostly, through step ~73,710 (1.5789 → 1.4632). Over the final ~5,500 steps (73,710 → 79,209) it stopped trending down: 1.4632 → 1.4830 → 1.4729 → 1.4959 → … → 1.4816, oscillating in a ~1.46–1.50 band rather than continuing to fall.
- **Did train and validation stay close?** Yes, consistently within ~0.01–0.03 of each other throughout — no sign of overfitting.
- **Any instability or spikes?** No large spikes; ordinary batch noise only.
- **Underfitting, good fit, or overfitting?** Good fit, transitioning toward a plateau. Not overfit (train/val still track together); not underfit (large loss drop, coherent samples). The final ~14% of steps show the same flattening signature that previously justified scaling the model up (Run 2 → Run 3).
- **Short interpretation:** this run made real further progress on Run 3 for its first ~86% of steps, then plateaued. The best checkpoint (1.4632), not the final one, is the correct number to quote for "how good is this model."

## 9. Sample Generations
*Generated fresh on 2026-09-11 from the best checkpoint, `checkpoints_run3b/step_73710.pt` — temperature 0.9, top_k 40, 120 new tokens.*

### Sample 1
- **Checkpoint used:** `step_73710.pt`
- **Prompt:** `"Once upon a time"`
- **Decoding settings:** temperature 0.9, top_k 40, max_new_tokens 120
- **Output:** "Once upon a time, there was a little girl named Lily. She loved to eat ice cream, especially the chocolate. One day, she was eating her ice cream when it suddenly disappeared! Lily looked around and saw that her ice cream had gone bad! She was very sad and worried because she didn't want anyone to eat it. Later that day, Lily's mom took her to a magical place where they could see magical people. Lily was very eager watering the plants, but she didn't know how to help the plants. She remembered that she had a bottle of water that she could use"
- **Assessment:** Fluent, grammatical, and internally consistent for roughly the first 40–50 tokens (the ice-cream scene), then the narrative loses its thread — "magical place," "magical people," and "watering the plants" appear with no causal connection to what came before. Classic long-range coherence decay for a model at this scale/budget.

### Sample 2
- **Checkpoint used:** `step_73710.pt`
- **Prompt:** `"The little dog looked at the ball and"`
- **Decoding settings:** temperature 0.9, top_k 40, max_new_tokens 120
- **Output:** "The little dog looked at the ball and wanted to play. But the ball was not a toy. It had sharp teeth. The dog wanted to protect the ball from the cat's cat. The cat was in danger. The dog thought of a plan to protect the ball. He ran away from the cat and the cat. The cat was scared, but the dog saved the ball. The cat was happy and said thank you to the dog. From that day on, the dog and the cat were good friends. Once upon a time, in a small town, there was a little boy named Tim. Tim had a big"
- **Assessment:** This is the strongest sample: a genuine conflict → resolution → moral arc (protect the ball, resolve the danger, become friends), grammatically clean throughout. It then correctly ends that story and begins a *new*, unrelated one ("Once upon a time... Tim"). This is the known, already-documented `generate()` limitation — the function doesn't stop at the EOS token, so it always fills the full token budget — not a new defect.

### Sample 3
- **Checkpoint used:** `step_73710.pt`
- **Prompt:** `"Tom and Sarah went to the park because"`
- **Decoding settings:** temperature 0.9, top_k 40, max_new_tokens 120
- **Output:** "Tom and Sarah went to the park because they saw a group of children playing. Suddenly, one of the children fell down and hurt his leg. Lily felt bad and went to pick it up. At the park, Lily and her friends tried to help the boy. They gave him some water and a bandage to put on his leg. The boy stopped being hurt and started playing with the ball. Lily noticed and asked the boy if he was okay. The boy said no, but Lily felt happy that she helped him. She realized that it's important to help others when they need it. Once upon a time, there was"
- **Assessment:** A real, previously unflagged failure mode: the prompt explicitly names "Tom and Sarah," but by the second sentence the model substitutes its own default protagonist name, "Lily," and never returns to the prompt's names at all. Otherwise grammatical and morally-coherent in TinyStories' typical style.

## 10. Qualitative Evaluation Checklist
- **Grammar / syntax:** Strong — no broken sentences in any of the 3 samples.
- **Prompt adherence:** Weak in one case — Sample 3 discarded the prompt's given character names entirely. Flagged as the main newly-observed weakness this report.
- **Character consistency:** Good *within* a single story once a name is established; not applicable across the story-to-story boundary.
- **Narrative coherence:** Good at short range (≲40–50 tokens); degrades over the full 120-token window — consistent with every prior report.
- **Repetition:** None observed.
- **Looping / degeneration:** None observed.
- **Long-range consistency:** Weak — wanders or switches to an unrelated story past ~50–100 tokens.
- **Domain fit:** Strong — matches TinyStories' simple vocabulary and moral-ending style throughout all 3 samples.
- **Overall output quality:** A clear, measurable loss/perplexity improvement over Run 3 at the best checkpoint, but the same long-standing qualitative failure modes persist — expected, since nothing in this run specifically targeted them.

## 11. Failure Modes
- **Observed issues:** (1) narrative coherence degrades past ~50 tokens; (2) a prompt supplying specific character names can be overridden by a high-frequency default name ("Lily") within the first sentence or two; (3) generation runs on into a second, unrelated story rather than stopping (pre-existing, documented `generate()` limitation).
- **When they appear:** (1) and (3) once generation exceeds roughly the length of one typical TinyStories example; (2) immediately, when the prompt supplies specific proper nouns.
- **Likely cause:** (1)/(3) — the model's effective coherence window is well under its 256-token `block_size`, consistent with its scale and training budget; (2) — "Lily" is very likely one of the most frequent protagonist names in TinyStories, giving the model a strong learned prior that a short prompt doesn't override.
- **Severity:** Low / expected for a model at this scale — none of these are regressions; all three are in the same category already logged in the Run 1/2/3 reports.
- **Examples:** Sample 3 above.

## 12. Comparison to Previous Runs
| Run | Steps | Train loss | Val loss | Val perplexity | Notes |
|---|---:|---:|---:|---:|---|
| **Run 3b (this run)** | 40,000 local / 79,209 cumulative | 1.4766 (final) / 1.4640 (best-ckpt) | 1.4816 (final) / **1.4632 (best, step 73,710)** | ≈4.40 (final) / **≈4.32 (best)** | 33.5M params, resumed from Run 3; improved through step 73,710, then flattened |
| Run 3 | 39,210 / 50,000 planned | 1.5582 | 1.5715 | ≈4.81 | 33.5M params, stopped by 5.5h time cap while still improving |
| Run 2 | 50,000 | 1.7263 | 1.7209 | ≈5.59 | 17M params, plateaued by ~step 30,000 |
| Run 1 | 5,000 | 2.4933 | 2.4926 | ≈12.09 | 17M params, ~9% of one epoch |

## 13. Interpretation
- **What improved in this run?** Validation loss dropped from 1.5715 (Run 3's endpoint) to 1.4632 (Run 3b's best) — a real further improvement at the same model scale, confirming Run 3's "still improving" signal was genuine rather than noise.
- **What did not improve?** The qualitative failure modes (Section 11) are unchanged from prior runs. The final ~5,500 steps of this run show no further improvement — oscillation, not decline.
- **Is the model still learning?** Not clearly, by the very end of this run — the last ~14% of steps oscillate rather than trend down. It was still learning through roughly the first 86%.
- **Is more training justified?** At the *same* model size, the evidence from this run's own tail is weak on its own — but Run 3b's middle section also had smaller flat patches that later broke through, so a short further continuation is the cheapest way to find out whether this is a genuine plateau or another temporary flat patch, before assuming a capacity increase is needed.
- **What is the main bottleneck right now: steps, model size, context length, or data?** Most likely model capacity again, on the same reasoning that applied to Run 2 → Run 3: loss is flattening well before the corpus is data-exhausted (cumulative tokens across Run 3 + Run 3b is only ≈1.40 epochs), so data volume is not yet the binding constraint.

## 14. Decision
- **Recommended next action:** Resume training from the best checkpoint (`checkpoints_run3b/step_73710.pt`, not the worse final checkpoint) for a further bounded run at the current 33.5M-parameter scale — this is the confirming experiment for whether Run 3b's tail-end flattening is a real plateau or a temporary flat patch, before committing to another capacity increase.
- **Why this is the best next action:** It's the cheapest way to get a clear answer — no architecture change, no new upcycling risk, reuses the exact code pattern already built for Run 3b. If it re-plateaus quickly, that's a fast, low-cost signal to scale the model up next (same logic as Run 2 → Run 3); if it keeps improving, more training alone is still the right lever.
- **Do not change yet:** model architecture/size, tokenizer, or dataset — none are implicated by this run's results yet.
- **Checkpoint to keep:** `checkpoints_run3b/step_73710.pt` — the canonical best Emere1.0 dense checkpoint to date.
- **Checkpoint to discard:** none need deleting; `step_79209_final.pt` is superseded as "the model to use" but is harmless to retain for the record.

## 15. Next Experiment Plan
- **Experiment name:** Run 3c — bounded continuation from `step_73710`
- **Single variable to change:** step count only, continuing training — resuming from the *best* checkpoint (73,710) rather than the run's final one (79,209), a deliberate small deviation from the Run 3 → Run 3b pattern, justified because Run 3b's own final checkpoint is demonstrably worse than an earlier one.
- **Hypothesis:** a further bounded number of steps from `step_73710` will either (a) push validation loss measurably below 1.4632 — supporting "keep training at this scale" — or (b) immediately re-plateau/oscillate in the same ~1.46–1.50 band — supporting "next move is a capacity increase," as with Run 2 → Run 3.
- **Success criteria:** a new best validation loss more than 0.01 below 1.4632 (i.e., a real improvement, not batch-to-batch noise) within the new step budget.
- **Metrics to monitor:** validation loss every 250 steps, specifically watching whether it sets a new low or continues oscillating around/above 1.4632.
- **Rollback condition:** if validation loss has not beaten 1.4632 by the halfway point of the new step budget with no clear downward trend, stop and treat that as confirmation the plateau is real — the next step becomes a capacity increase, not further steps at this scale.

## 16. Appendix

### Commands used
```bash
# Sample generation for Section 9 (2026-09-11), using a temporary script
# since sample.py's default Config doesn't match Run 3/3b's architecture:
python _tmp_sample_run3b.py

# Resume training (Run 3c, launched immediately after this report):
python train_run3c.py
```

### File paths
```text
Best checkpoint:      checkpoints_run3b/step_73710.pt
Final checkpoint:     checkpoints_run3b/step_79209_final.pt
Training log:         train_run3b.log
Resume script used:   train_run3b.py
New run script:       train_run3c.py
New checkpoint dir:   checkpoints_run3c/
```

### Extra notes
- This report was written ~6 weeks after the run completed, entirely from preserved logs and checkpoints — no information was lost, but there was no same-day operator commentary beyond what's already in the emere1-slm memory record.
- Run 3c was launched on 2026-09-11 immediately following this report, per the Decision above.

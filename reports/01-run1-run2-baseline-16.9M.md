---
project: Emere1.0
type: Evaluation Report
date: 2026-07-29
covers: Run 1 (5,000 steps) and Run 2 (50,000 steps)
---

# SLM Training Evaluation Report

Two reports, one per completed dense training run, using the Perplexity-recommended structure. All figures below are pulled directly from training logs and checkpoint files, not estimated — where a data point genuinely isn't available (e.g. run 1's full per-step curve, which wasn't preserved to disk), that's stated rather than filled in.

---

# Report A — Run 1 (5,000 steps)

## 1. Project Summary
- **Model name:** Emere1.0 (TinyGPT)
- **Training goal:** first genuine from-scratch training run — validate the full pipeline (tokenizer → encoding → training → sampling) works end to end and produces coherent output
- **Dataset:** TinyStories (roneneldan/TinyStories)
- **Training date:** 2026-07-25/26
- **Hardware:** Apple M4 Pro, 14 cores (10P/4E), 48GB unified memory, PyTorch MPS (Metal) backend

## 2. Dataset
- **Source:** [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (HuggingFace)
- **Raw size:** 1.8GB train.txt, 18MB val.txt
- **Tokenized size:** 859MB train.bin, 8.6MB val.bin (uint16 token arrays)
- **Train/val/test split:** HuggingFace's own predefined train (2,119,719 stories) / validation (21,990 stories) split. **No held-out test set** — only train/val exist for this project.
- **Tokenizer type:** SentencePiece BPE, trained from scratch on the TinyStories train split (not reused from elsewhere)
- **Vocabulary size:** 8,000

## 3. Model Configuration
- **Architecture:** decoder-only GPT-style transformer, written from scratch in PyTorch (manual QKV attention, no library transformer blocks)
- **Layers:** 6
- **Heads:** 6
- **Embedding size:** 384
- **Context length:** 256 tokens
- **Parameter count:** ~17M
- **Initialization details:** PyTorch defaults throughout (`nn.Linear` Kaiming-uniform, `nn.Embedding` normal) — no custom/scaled residual initialization (e.g. GPT-2's 1/√(2·n_layer) output-projection scaling). Worth flagging as a real gap, though likely low-impact at only 6 layers deep.

## 4. Training Setup
- **Framework:** PyTorch 2.13.0
- **Optimizer:** AdamW, weight_decay 0.01
- **Learning rate:** 3e-4 peak
- **LR schedule:** cosine decay to 10% of peak, 200-step linear warmup
- **Batch size:** 32
- **Gradient clipping:** 1.0 (global norm)
- **Precision:** fp32 (no autocast/mixed precision used)
- **Checkpoint interval:** every 250 steps
- **Total steps planned:** 5,000
- **Steps completed:** 5,000

## 5. Training Results
- **Final train loss:** 2.4933
- **Final validation loss:** 2.4926
- **Train perplexity:** ≈ 12.10
- **Validation perplexity:** ≈ 12.09
- **Best checkpoint:** `step_4999.pt` (final) — the only data point available; **this run's per-step log wasn't persisted to disk**, so an intermediate minimum can't be confirmed. Loss was reported as still decreasing steadily at the final checkpoint (no plateau observed), based on the recorded trajectory from a step-0 baseline of ~9.14.
- **Training time:** ~27 minutes
- **Tokens processed:** 40,960,000 (5,000 × 32 × 256) — **~9.1% of one epoch** over the 450.4M-token train corpus

## 6. Learning Curve Notes
- **Did train and val loss track closely?** Yes — gap of ~0.0007 at the final step, essentially identical.
- **Did validation loss diverge?** No evidence of divergence.
- **Any instability or spikes?** None visible in the recorded checkpoints; note the caveat above that the full per-step curve wasn't saved for this run.
- **Underfit / well-fit / overfit?** Underfit relative to available data — the model had seen under 10% of one epoch when the run ended, well short of exhausting the corpus.

## 7. Qualitative Samples

### Prompt 1
- **Prompt:** "Once upon a time"
- **Output:** *"Once upon a time, there was a little girl named Lily. She had a big doll that she loved very much. One day, she saw a doll with green eyes. Lily asked her mom if she could draw a picture of her doll. Her mom said yes, and Lily was happy to learn new things, but also happy because she found a purple dress that had a new doll. Then, Lily's mom said, 'Wow, that's a great dress, that's a great dress, sweetheart.' They played with the dress and had the dress. Lily was happy that her mom had a fun"*
- **Notes:** Grammatical throughout, consistent character naming. Degrades into verbatim repetition ("that's a great dress, that's a great dress") and loosening logic near the end of the 120-token generation.

### Prompt 2
- **Prompt:** "The little dog"
- **Output:** *"The little dog was scared and didn't know what to do. He was so scared he could not go on an adventure. The big dog tried to run away, but it was too late. The little dog was just sad, but he learned his lesson and never felt sad again. Once upon a time, there was a little bunny named Benny. Benny had a big, heavy cat named Fluffy..."*
- **Notes:** First story resolves cleanly (scared → lesson learned). Then the model abruptly launches into a second, unrelated story ("Once upon a time... Benny...") with no transition — see Error Analysis for the likely root cause.

## 8. Error Analysis
- **Repetition:** Present — verbatim phrase looping observed in Prompt 1's output near generation end.
- **Coherence:** Strong at sentence level; weakens over the course of a full generation.
- **Hallucination:** Minor semantic slips (e.g. "big, heavy cat" is an odd descriptor) rather than factual errors — expected in a fictional-story domain.
- **Long-range consistency:** Breaks down by ~100+ tokens — most visibly as an unmarked switch into an entirely new, unrelated story (Prompt 2).
- **Failure modes:** (1) end-of-generation repetition loops; (2) abrupt unmarked story-switching. On (2): `prepare_data.py` joins consecutive TinyStories examples with only a single newline, no document/EOS separator — the model was never given a training signal for "this is where a story ends," so it has no learned boundary to prefer over just continuing into new content. This is a data-prep detail, not a capacity issue, and reproduces in Run 2 as well (see below).

## 9. Decision
- Continue training on the same setup.
- Recheck validation loss at 10k, 25k, and 50k steps.
- Keep the same dataset and tokenizer for now.
- Only consider increasing model size if validation loss plateaus.

## 10. Next Actions
- **Immediate next step:** run 50,000 steps on the identical architecture, dataset, and tokenizer (~91% of one epoch vs. this run's ~9%).
- **Hypothesis to test:** whether loss keeps improving meaningfully with ~10x more steps on the same unique data, or plateaus before reaching a full epoch.
- **Metrics to watch:** train/val gap (memorization signal), val loss trend at 10k/25k/50k checkpoints, whether repetition loops persist in longer qualitative samples.

---

# Report B — Run 2 (50,000 steps)

## 1. Project Summary
- **Model name:** Emere1.0 (TinyGPT)
- **Training goal:** test whether the diminishing-returns question from Run 1 holds — does loss keep improving with ~10x more steps on the same data, or plateau
- **Dataset:** TinyStories (roneneldan/TinyStories) — identical to Run 1
- **Training date:** 2026-07-29
- **Hardware:** Apple M4 Pro, 14 cores (10P/4E), 48GB unified memory, PyTorch MPS (Metal) backend

## 2. Dataset
- **Source:** [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (HuggingFace) — unchanged from Run 1
- **Raw size:** 1.8GB train.txt, 18MB val.txt
- **Tokenized size:** 859MB train.bin, 8.6MB val.bin
- **Train/val/test split:** same as Run 1 — HuggingFace train/val split, no held-out test set
- **Tokenizer type:** same SentencePiece BPE tokenizer as Run 1, reused (not retrained)
- **Vocabulary size:** 8,000

## 3. Model Configuration
Identical to Run 1 — same architecture, no changes:
- **Architecture:** decoder-only GPT-style transformer, from scratch in PyTorch
- **Layers:** 6 · **Heads:** 6 · **Embedding size:** 384 · **Context length:** 256 tokens
- **Parameter count:** ~17M
- **Initialization details:** PyTorch defaults (same caveat as Run 1 — no scaled residual init)

## 4. Training Setup
Identical to Run 1 except step count:
- **Framework:** PyTorch 2.13.0 · **Optimizer:** AdamW (weight_decay 0.01) · **Learning rate:** 3e-4 peak, cosine schedule, 200-step warmup · **Batch size:** 32 · **Gradient clipping:** 1.0 · **Precision:** fp32
- **Checkpoint interval:** every 250 steps (202 checkpoints saved, ~13GB total)
- **Total steps planned:** 50,000
- **Steps completed:** 50,000

## 5. Training Results
- **Final train loss:** 1.7263
- **Final validation loss:** 1.7209
- **Train perplexity:** ≈ 5.62
- **Validation perplexity:** ≈ 5.59
- **Best checkpoint:** `step_48250.pt` — val loss 1.7062 (perplexity ≈ 5.51), the true minimum across the full logged curve. Essentially tied with the final checkpoint (within normal eval-batch noise), not a meaningfully different model.
- **Training time:** 3h 46m (19:16 → 23:02)
- **Tokens processed:** 409,600,000 (50,000 × 32 × 256) — **~90.9% of one epoch**

## 6. Learning Curve Notes
- **Did train and val loss track closely?** Yes throughout — gap stayed small (~0.005–0.03 at individual eval points, some noise-driven crossovers where val briefly read below train, e.g. step 39,750). No systematic divergence.
- **Did validation loss diverge?** No.
- **Any instability or spikes?** No sharp spikes; normal eval-to-eval noise (eval used a 50-batch sample, not the full val set, so ±0.02–0.03 jitter between adjacent checkpoints is expected).
- **Underfit / well-fit / overfit?** Well-fit, transitioning toward a plateau. Loss curve: 9.14 (step 0) → 2.74 (step 2,250) → 1.78 (step 29,750) → 1.72 (step 49,999). **Roughly 90% of the total loss reduction happened in the first 60% of steps** — the last ~20,000 steps bought only ~0.06 of further improvement. Not overfit (val never rises above train meaningfully), but clearly diminishing returns on "more of the same data" at this model size.

## 7. Qualitative Samples

### Prompt 1
- **Prompt:** "Once upon a time"
- **Output:** *"Once upon a time there was a girl named Jane. Jane loved to read books. Every day she would read the stories of the people she would read. One day, she decided that she wanted to hear a story about a princess and a dragon. So Jane went to find the dragon. She looked under the curtains, behind the curtains, and even in the closet. But the dragon looked still, like a real princess. Jane was very sad. She didn't know what to do. Then, Jane had an idea! She went to the store and bought some glue because she could make a special story to tell"*
- **Notes:** Sustains a single plot arc (goal → search → idea) across the full 120 tokens without collapsing into repetition, unlike Run 1's Prompt 1. One semantic non-sequitur ("the dragon looked still, like a real princess") — logic holds better than Run 1 but isn't perfect.

### Prompt 2
- **Prompt:** "The little dog"
- **Output:** *"The little dog was so happy to have a new friend. They played together until the sun went down, and the little dog went home, feeling sleepy. Once upon a time, in a small village, there lived a little girl named Lily. Lily had a red ball that she loved very much. One day, she found a big piece of paper on the ground. She was very happy and wanted to keep the paper. Lily called her friend, Tom. 'Tom, come home! It's not safe to make a paper with the paper!' Tom heard Lily's cry and came to pick her up"*
- **Notes:** Same abrupt story-switch pattern as Run 1 ("...went home, feeling sleepy. Once upon a time... Lily..."), confirming it's a data artifact, not something more training fixed. Minor incoherent dialogue line ("make a paper with the paper").

## 8. Error Analysis
- **Repetition:** Not observed in either sample — a real improvement over Run 1's verbatim phrase-looping.
- **Coherence:** Noticeably stronger — sustains a full goal→action→resolution arc within a single story, where Run 1 tended to lose the thread.
- **Hallucination:** Still present as minor semantic non-sequiturs (dragon/princess line, "paper with the paper" line) — not resolved by more training, since this is a model-capacity ceiling issue, not a data-volume one.
- **Long-range consistency:** Same unmarked story-switch failure mode as Run 1, at almost the identical token position in both prompt-2 samples — strong evidence this is the missing-EOS-separator data issue identified in Report A, and it will persist regardless of further steps on this dataset until that's fixed.
- **Failure modes:** (1) unmarked story-switching (confirmed data-prep root cause, unresolved by 10x more steps — as expected, since more steps can't fix a missing training signal); (2) residual local hallucination/non-sequiturs, consistent with a 17M-parameter ceiling rather than a training-duration issue.

## 9. Decision
- **Do not continue the dense line further on the same architecture and data** — clear diminishing returns after ~step 30,000 (2.74→1.78 in the first 30k steps vs. only 1.78→1.72 in the last 20k).
- **Model capacity (17M params), not remaining data, is now the likely binding constraint** — the run covered ~91% of one epoch and val loss has flattened; more of the same data is unlikely to move the needle further.
- **Fix the missing document/EOS-separator before any future dense-line run** — this is a concrete, fixable data-prep bug (confirmed root cause of the story-switching failure mode in both runs), not something more steps or a bigger model would solve on its own.
- **Proceed with the already-approved next stage** (MoE upcycling from this checkpoint onto the vault lobes) rather than a third dense run on TinyStories alone — this is the actual decision made and already in progress as of this report.

## 10. Next Actions
- **Immediate next step:** MoE stage 3 training (in progress) — 8 experts upcycled from `step_49999.pt`, one per vault lobe + Private, router-supervised on source lobe.
- **Hypothesis to test:** whether a larger model (more layers/embedding width) meaningfully improves loss beyond this plateau, if a future dense-only run is ever revisited — not yet tested.
- **Metrics to watch:** MoE per-lobe eval loss (especially the smallest lobes — Hippocampus at 890 tokens, Frontal Lobe at 5,493 — where memorization vs. any generalization is the open question); whether fixing the EOS-separator gap in a future dense run reduces the story-switching failure mode directly.

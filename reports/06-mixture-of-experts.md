---
project: Emere1.0
type: Evaluation Report
template: slm_training_evaluation_template.md
run: MoE Stage 3 (Vault Lobe Mirroring)
date: 2026-07-29
---

# SLM Training Evaluation Report — MoE Stage 3 (Vault Lobe Mirroring)

## 1. Run Summary
- **Run name:** MoE Stage 3 — Vault Lobe Mirroring
- **Date started:** 2026-07-29 23:04:09 BST
- **Date completed:** 2026-07-29 23:58:36 BST
- **Status:** Complete — reached full planned step count (no early stop needed)
- **Objective:** mirror the vault's 7-lobe + Private structure as separate MoE experts, upcycled from the Run 2 dense checkpoint. Primary open question going in: does the router generalize at inference (unlabeled prompts) or collapse to always picking the largest lobe (Temporal Lobe, 353K of ~658K total tokens)?
- **Author / operator:** Claude Code, autonomous build + training, Founder-approved (explicit "proceed with MoE" instruction, scoped to this SLM only, no other agent/LLM touching Private content)
- **Notes:** first-ever MoE result for this project — no prior MoE run to compare against.

## 2. Environment
- **Project folder:** `./`
- **Git commit / version:** not a git repo yet
- **Python version:** 3.11.3
- **Framework versions:** PyTorch 2.13.0, sentencepiece 0.2.2
- **Hardware:** Apple M4 Pro, 14 cores (10P/4E), 48GB unified memory
- **OS:** macOS (Darwin)
- **Device used for training:** MPS (Metal GPU)
- **Mixed precision enabled:** No (fp32 throughout)
- **Checkpoint directory:** `checkpoints_moe/`

## 3. Dataset
- **Dataset name:** Vault lobes (Brainstem, Cerebellum, Frontal Lobe, Hippocampus, Occipital Lobe, Parietal Lobe, Temporal Lobe) + Private
- **Dataset source:** `<private notes vault>/` — direct mechanical file read, no LLM in the loop
- **Domain / content type:** personal knowledge base / project notes / private notes — not fiction, a hard domain shift from the TinyStories base
- **Raw dataset size:** ~3.4MB combined across all lobes (Brainstem 13KB, Frontal Lobe 12KB, Hippocampus 2.3KB, Occipital Lobe 175KB, Parietal Lobe 382KB, Temporal Lobe 606KB, Private 20KB; Cerebellum 0)
- **Cleaned dataset size:** n/a — raw markdown used directly, no cleaning pass
- **Train split:** all available tokens per lobe, sampled uniformly across lobes per batch (not proportional to lobe size) — no held-out split
- **Validation split:** same pool as train, sampled with the same `get_batch()` — **not a true held-out set**, a real limitation given the data volume (see Loss Curve Assessment)
- **Test split:** none
- **Estimated total tokens:** 658,511 across the 7 usable lobes (Brainstem 6,885 / Frontal Lobe 5,493 / Hippocampus 890 / Occipital Lobe 80,922 / Parietal Lobe 204,135 / Temporal Lobe 353,252 / Private 6,934). Cerebellum: 0, excluded.
- **Data quality notes:** severe size imbalance by design (mirrors the real vault, not engineered for balance). 4 of 8 lobes (Cerebellum, Hippocampus, Frontal Lobe, Brainstem) have too little text for genuine generalization — expected to mostly memorize their handful of source documents, not learn general "lobe style."
- **License / usage notes:** private personal data. Private scope explicitly narrowed — SLM training only, no other agent/LLM, nothing pasted to chat or pushed elsewhere.

## 4. Tokenizer
- **Tokenizer type:** SentencePiece BPE
- **Tokenizer training corpus:** TinyStories (reused from the dense stage — not retrained on vault content, these corpora are too small to train a tokenizer on their own)
- **Vocabulary size:** 8,000
- **Special tokens:** `pad_id=0, unk_id=1, bos_id=2, eos_id=3` (reserved at tokenizer training time)
- **Tokenizer file path:** `tokenizer/tinyslm.model`
- **Encoding format:** uint16 token arrays, one `.bin` per lobe (`data/vault_processed/<lobe>.bin`)
- **Tokenizer notes:** vault vocabulary (proper nouns, technical terms, project names) is a domain mismatch for a tokenizer trained purely on TinyStories' children's-story vocabulary — expect more byte-fallback fragmentation than the dense stage saw. Visible in the qualitative samples below (broken words like "outheads," "Realshal").

## 5. Model Configuration
- **Architecture type:** decoder-only GPT-style transformer with Mixture-of-Experts MLP layers, written from scratch
- **Model class / file:** `TinyMoEGPT`, `slm/moe_model.py`
- **Parameter count:** ~66.5M (shared backbone ~9.9M + 8 experts × 6 layers × ~1.18M each ≈ 56.6M + router)
- **Number of layers:** 6 (unchanged from dense base)
- **Number of attention heads:** 6 (unchanged)
- **Embedding dimension:** 384 (unchanged)
- **Context length:** 256 tokens (unchanged)
- **Dropout:** 0.1
- **Activation function:** GELU (expert MLPs)
- **Positional encoding:** learned absolute positional embeddings (unchanged)
- **Weight tying enabled:** No
- **Initialization notes:** **not random init** — sparse upcycling from Run 2's `step_49999.pt`: embeddings, attention, layer norms, and LM head copied directly; the single dense MLP cloned into all 8 experts as a warm start. Only the router (`nn.Linear(n_embd, n_experts)` per layer) is genuinely randomly initialized — everything else starts from a fully-trained 50,000-step model.

## 6. Training Configuration
- **Total planned steps:** 3,000
- **Completed steps:** 3,000 (100%)
- **Batch size:** 32
- **Gradient accumulation:** none
- **Effective batch size:** 32
- **Tokens per step:** 8,192 (32 × 256)
- **Optimizer:** AdamW, weight_decay 0.01
- **Learning rate:** 1e-4 peak (lower than the dense stage's 3e-4 — fine-tuning an already-trained backbone, not learning from scratch)
- **Learning rate schedule:** cosine decay to 10% of peak
- **Warmup steps:** 100
- **Weight decay:** 0.01
- **Gradient clipping:** 1.0 (global norm)
- **Evaluation interval:** every 100 steps, 20 batches per eval
- **Checkpoint interval:** every 100 steps
- **Random seed:** 42
- **Resume from checkpoint:** yes — upcycled from `checkpoints/step_49999.pt` (Run 2), not a resume in the literal sense (architecture changed) but a weight-transfer warm start

## 7. Training Results
| Metric | Value | Notes |
|---|---:|---|
| Final train loss | 0.6751 | single-batch value, noisy |
| Final validation loss | 0.4329 | 20-batch average |
| Best validation loss | 0.4159 (step 2900) | essentially tied with final — no clear winner |
| Train perplexity | n/a | see note below |
| Validation perplexity | n/a | see note below |
| Total runtime | ~54.5 minutes | 23:04:09 → 23:58:36 |
| Tokens processed | 24,576,000 (3,000 × 32 × 256) | far exceeds the 658K-token pool many times over — see Loss Curve Assessment |
| Approx. epochs completed | not meaningful | lobe-balanced sampling with replacement, not epoch-based; smallest lobes (Hippocampus, 890 tokens) were resampled hundreds of times |
| Best checkpoint | `step_2900.pt` (practically tied with `step_2999.pt`) | |

**Note on perplexity:** the logged loss is a *combined* value — LM cross-entropy + 0.1× router-supervision cross-entropy (see `train_moe.py`) — not pure next-token loss. Converting it directly to perplexity would be misleading (it would understate true LM perplexity, since part of the loss is the router's classification task, not next-token prediction). `train_moe.py` doesn't currently log the two components separately — a real gap, worth fixing if precise perplexity comparison to the dense runs is wanted later.

## 8. Loss Curve Assessment
- **Did train loss decrease steadily?** Sharp initial drop (7.47 → ~0.5 by step 1,600), then noisy plateau for the remaining ~1,400 steps — no further improvement.
- **Did validation loss decrease steadily?** Same shape: 7.53 → 0.41–0.52 range by step 1,600, then flat with noise (fluctuating between 0.42 and 0.52 through the end, no clear trend).
- **Did train and validation stay close?** Reasonably, but this comparison is weaker evidence than in the dense runs — the "validation" set is drawn from the *same* pool as training via the same `get_batch()` sampling, not a genuinely held-out split. For a dataset this size (658K tokens, many lobes resampled hundreds of times over 3,000 steps), train/val closeness here is a much weaker signal against memorization than in the TinyStories runs.
- **Any instability or spikes?** No sharp spikes, but real batch-to-batch noise (0.42–0.99 range in single-batch train loss) — expected given each batch mixes lobes of wildly different sizes.
- **Evidence of underfitting, good fit, or overfitting?** Converged and plateaued by ~step 1,600 — the remaining ~1,400 steps were likely unnecessary. Given the smallest lobes (Hippocampus, Frontal Lobe, Brainstem) have only hundreds to low-thousands of tokens, **some memorization of those lobes' source documents is expected and, per the original design intent, largely acceptable** (see Founder's stated intent: faithful recall over generalization for private notes like the Private section).
- **Short interpretation:** fine-tuning saturated fast on a tiny, imbalanced dataset — 3,000 steps was more than sufficient; a shorter run (~1,500–2,000 steps) would likely have reached the same result with less wasted compute.

## 9. Sample Generations
*Withheld from the public copy. This model was fine-tuned on private personal notes, and its samples reproduce that content (see the note on memorisation in section 8). The original report holds four prompts across eight routing conditions.*

## 10. Qualitative Evaluation Checklist
- **Grammar / syntax:** degrades sharply on the smallest lobes (Brainstem, Frontal Lobe) into broken/fragmented tokens; holds up reasonably on Temporal Lobe and the free-router choices.
- **Prompt adherence:** generally weak to moderate — the model quickly drifts into lobe-flavored content regardless of what the prompt asked, more so than the dense TinyStories model did.
- **Character consistency:** not applicable in the same sense as fiction — the relevant analogue (thematic consistency within a lobe) holds well for Temporal Lobe, moderately for Private, poorly for the smallest lobes.
- **Narrative coherence:** low — this domain isn't narrative, and none of the lobes produce sustained coherent prose the way the TinyStories dense model does.
- **Repetition:** present and severe in the smallest/most overfit lobes — Brainstem's forced output shows literal token-loop degeneration ("Realshal: Realshal: Realshal:").
- **Looping / degeneration:** yes, in Brainstem and (to a lesser extent) Parietal Lobe.
- **Long-range consistency:** not meaningfully testable at this domain/scale — outputs are short, fragmentary phrase-salad more often than sustained passages.
- **Domain fit:** strong for Temporal Lobe (log/alert register) and Private (private notes); weak for the smallest lobes; the free router's domain fit is the most encouraging result — it tracked prompt content across all 4 test prompts rather than defaulting to one lobe.
- **Overall output quality:** well below the dense TinyStories model's fluency (expected — 658K tokens of imbalanced, non-narrative domain data vs. 450M+ tokens of a purpose-built children's-story corpus), but the *architectural* result (differentiated, prompt-sensitive routing) is a genuine success against the question this run was designed to answer.

## 11. Failure Modes
- **Observed issues:** (1) token-level garbling and repetition loops in the smallest lobes (Cerebellum unused entirely, Hippocampus/Frontal Lobe/Brainstem near-memorization with degeneration); (2) weak prompt adherence — forced-lobe generations drift into lobe-themed content fast regardless of the prompt; (3) validation loss isn't a trustworthy held-out signal given the sampling design.
- **When they appear:** immediately in any forced-lobe generation from a sub-10K-token lobe; router free-choice degenerates less, staying closer to the base model's fluency.
- **Likely cause:** insufficient data per small lobe (inherent to the vault's actual size, not a training bug), combined with only 3,000 fine-tune steps on an already-converged loss curve (more steps wouldn't fix this — it's a data-volume ceiling, confirmed by the flat loss curve after step 1,600).
- **Severity:** moderate for the small lobes' output quality; low for the core question this run was designed to test (does routing generalize) — that result looks solid.
- **Examples:** see Section 9, Samples 2 and 3, and the Brainstem repetition example above.

## 12. Comparison to Previous Runs
| Run | Steps | Train loss | Val loss | Val perplexity | Notes |
|---|---:|---:|---:|---:|---|
| MoE Stage 3 (this run) | 3,000 | 0.6751 | 0.4329 | n/a (combined LM+router loss, not directly comparable) | first MoE run, no prior MoE baseline |
| Run 2 (dense, upcycle source) | 50,000 | 1.7263 | 1.7209 | ≈5.59 | TinyStories only; this run's starting point |
| Run 1 (dense) | 5,000 | 2.4933 | 2.4926 | ≈12.09 | TinyStories only |

Not a fair like-for-like comparison — different domain, different loss composition, different data scale. Included for the record per the standing template, not as a ranking.

## 13. Interpretation
- **What improved in this run?** The open question from the prior session — does the router generalize or collapse to the biggest lobe — got a real, positive answer: it tracked prompt content across all 4 test prompts (identity/verification theme → Brainstem-adjacent register; project-status theme → Frontal-Lobe-adjacent register) rather than defaulting to Temporal Lobe every time.
- **What did not improve?** Output quality/fluency in the smallest lobes remains poor — inherent to how little source text exists, not something this run's design could have fixed.
- **Is the model still learning?** No — loss plateaued by step ~1,600 of 3,000. Further steps on this exact data/setup wouldn't help.
- **Is more training justified?** Not on the same data. More vault content (if it grows over time) would be the lever, not more steps on the current ~658K-token pool.
- **What is the main bottleneck right now?** Data volume per lobe, unambiguously — not steps, not model size (this reused the dense backbone's proven capacity), not the tokenizer (though tokenizer mismatch is a secondary contributor to the garbling).

## 14. Decision
- **Recommended next action:** treat this MoE run as a successful proof of concept for the routing mechanism, not yet a production-quality generator. Don't invest in more MoE training on the current lobe data — it's already converged.
- **Why this is the best next action:** the loss curve and qualitative samples both point to data volume, not training configuration, as the limiter. More steps or hyperparameter tuning won't move this forward.
- **Do not change yet:** the upcycling approach, the router-supervision design, and the lobe-uniform batch sampling all worked as intended — no evidence any of these need revisiting.
- **Checkpoint to keep:** `step_2900.pt` or `step_2999.pt` (practically identical).
- **Checkpoint to discard:** none needed — total footprint is manageable; keep all for now given this is the first MoE run and a useful reference point.

## 15. Next Experiment Plan
- **Experiment name:** MoE decomposed-loss logging (small fix, not a new training run)
- **Single variable to change:** log LM loss and router loss separately in `train_moe.py` instead of only the combined value.
- **Hypothesis:** the combined loss is masking a possibly-worse-than-it-looks pure LM loss, given the domain shift; separating them would make future MoE evaluation reports comparable to the dense runs' perplexity figures.
- **Success criteria:** next MoE report can state a real, comparable perplexity number instead of "n/a."
- **Metrics to monitor:** LM-only loss trend vs. router-only loss trend, checked independently for a plateau.
- **Rollback condition:** n/a — this is a logging change, not a training change; no risk to existing checkpoints.

## 16. Appendix

### Commands used
```bash
python3 prepare_vault_data.py
python3 encode_vault_data.py
python3 train_moe.py
python3 evaluate_moe.py
```

### File paths
```text
slm/moe_model.py           — architecture
prepare_vault_data.py      — vault → per-lobe raw text
encode_vault_data.py       — per-lobe raw text → token .bin + meta.json
train_moe.py                — upcycling + fine-tune training loop
evaluate_moe.py             — forced-lobe / free-router sample generation
checkpoints_moe/step_2999.pt — final checkpoint
moe_evaluation_output.txt   — full sample generations (4 prompts × 8 conditions)
```

### Extra notes
- Private scope reminder: this run's data pipeline is mechanical file I/O only — no LLM (including Claude, including this report-writing pass) read raw Private file contents beyond what's already been surfaced in prior conversation turns for architecture/sizing purposes. Sample outputs quoted above are the *model's own generated text*, not reproductions of source documents.

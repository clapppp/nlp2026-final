# PART-II Plan: Paraphrase Detection

This note is the implementation reference for the Part II work in this repo.
The scope is **Paraphrase Detection only**. Do not spend Part II development
time on `sonnet_generation.py` unless the scope is explicitly changed.

## Source Of Truth

- Assignment details: `nlp2026-final-프로젝트_세부_안내_v1.pdf`
- Starter code: `paraphrase_detection.py`, `datasets.py`, `evaluation.py`,
  and `models/gpt2.py`
- Part I dependencies: GPT-2 layers, attention, and `AdamW` must keep working
  because the paraphrase model reuses them.

## Assignment Contract

### Task

- Fine-tune GPT-2 on Quora question pairs to decide whether two questions are
  paraphrases.
- Treat the task as a **cloze-style next-token task**, not only as a generic
  binary classifier.
- Prompt GPT-2 with a question about the pair and make the answer correspond to
  the word `yes` or `no`.

A canonical prompt is:

```text
Is "{sentence1}" a paraphrase of "{sentence2}"? Answer "yes" or "no":
```

Keep train, dev, and test prompt construction aligned. Prompt variants are valid
experiments, but each experiment should use one consistent prompt policy across
splits.

### Data And Evaluation

- Use the provided Quora splits in `data/`.
- PDF split sizes:
  - train: 141,506 examples
  - dev: 20,215 examples
  - test: 40,431 examples
- Train, tune, and evaluate only with the provided train and dev splits.
- Do not use official test labels, external test data, or manual CSV edits to
  tune the model.
- The assignment evaluates Paraphrase Detection with accuracy. Keep dev F1 in
  logs as a useful diagnostic, but optimize against the assignment contract.

### Expected Output

- The entrypoint is `python paraphrase_detection.py --use_gpu` for a full GPU
  training and prediction run.
- `paraphrase_detection.test` writes prediction files for dev and test:
  - `predictions/para-dev-output.csv`
  - `predictions/para-test-output.csv`
- Output predictions should remain binary paraphrase labels compatible with the
  provided evaluation and submission pipeline.

## Repo Map For This Task

| File | Role In Paraphrase Work |
| --- | --- |
| `paraphrase_detection.py` | `ParaphraseGPT`, training loop, checkpointing, prediction writing |
| `datasets.py` | Quora loading, preprocessing, prompt tokenization, batch collation |
| `evaluation.py` | Dev/test paraphrase evaluation and `argmax` prediction conversion |
| `models/gpt2.py` | GPT-2 outputs `last_token` and maps hidden states to vocabulary logits |
| `optimizer.py` | `AdamW` used by the training loop |

## Current Starter Code Contract

Read these contracts before changing the implementation:

1. `GPT2Model.forward(...)` returns a dictionary with:
   - `last_hidden_state`: token-level hidden states
   - `last_token`: hidden state for the last non-padding prompt token
2. `GPT2Model.hidden_state_to_token(...)` maps hidden states to vocabulary
   logits through the tied GPT-2 token embedding matrix.
3. `evaluation.model_eval_paraphrase(...)` currently expects model output that
   can be reduced with `argmax(axis=1)` to binary predictions.
4. The cleanest output contract for `ParaphraseGPT.forward(...)` is therefore:

```python
logits.shape == (batch_size, 2)
```

Use this class order consistently:

```text
class 0 -> no
class 1 -> yes
```

## Recommended Implementation Direction

Prefer a cloze verbalizer implementation while preserving the binary output
contract expected by the rest of the repo:

1. Build a cloze prompt for each question pair.
2. Run the prompt through `GPT2Model`.
3. Take the final prompt hidden state from `output["last_token"]`.
4. Convert that hidden state to vocabulary logits with
   `GPT2Model.hidden_state_to_token(...)`.
5. Select the GPT-2 logits for the `no` and `yes` answer tokens.
6. Return the selected logits ordered as `[no_logit, yes_logit]`.
7. Train with Quora labels `0` and `1` against those two logits.

This keeps the model behavior close to the PDF description: the final decision
comes from the next-token preference for `no` versus `yes`.

Important tokenizer note:

- The starter comment mentions BPE token IDs for `yes` and `no`.
- Verify the exact token IDs with the same GPT-2 tokenizer and the chosen prompt
  suffix before locking the verbalizer.
- Prompt trailing spaces and GPT-2 BPE whitespace behavior can change which
  answer token is appropriate.

Fallback baseline:

- The provided starter class includes `paraphrase_detection_head =
  nn.Linear(hidden_size, 2)`.
- A binary linear head over `last_token` is a valid baseline to compare against,
  but the verbalizer path above is the first implementation to try for this
  cloze-style assignment.

## Starter Code Issues To Resolve First

The current repository state has paraphrase-specific inconsistencies that should
be resolved before a serious training run:

1. `ParaphraseGPT.forward(...)` is still a TODO in
   `paraphrase_detection.py`.
2. `ParaphraseDetectionDataset.collate_fn(...)` currently tokenizes label words
   such as `yes` and `no`, while the training loop and evaluation path are
   shaped like a two-class loss and `argmax` pipeline.
3. Train/dev prompt construction and test prompt construction currently use
   different templates.
4. The train/dev prompt string in `datasets.py` should be checked for quote and
   suffix consistency before experiments.

Resolve the label policy explicitly:

- Recommended policy: keep dataset labels as `torch.long` class IDs `0` or `1`
  and return two answer logits from the model.
- Alternative policy: train over full vocabulary token IDs, but then update the
  evaluation and prediction conversion carefully so generated `yes` and `no`
  still map back to the required binary outputs. Do not mix this policy with a
  two-logit loss.

## Implementation Checklist

### 1. Stabilize Inputs

- Centralize or mirror one prompt template for train, dev, and test.
- Keep `sent_ids` intact through collation so saved predictions remain aligned.
- Keep label tensors numeric and shape-compatible with the chosen loss.
- Check tokenization lengths and truncation behavior on long Quora pairs.

### 2. Implement Model Output

- Implement `ParaphraseGPT.forward(...)`.
- Preserve attention masks so `last_token` is the final non-padding prompt
  token.
- Return two logits in the documented class order.
- Remove or clearly isolate unused task heads after deciding whether the
  verbalizer or linear-head baseline is active.

### 3. Train And Checkpoint

- Start with the existing training loop and best-dev-accuracy checkpointing.
- Record:
  - model size
  - prompt template
  - answer-token policy
  - learning rate
  - batch size
  - epochs
  - dev accuracy and dev F1
- Use GPU for full fine-tuning runs. Use a one-batch smoke test before spending
  time on a full dataset run.

### 4. Predict

- Load the best checkpoint.
- Evaluate dev with the same prompt and label contract used in training.
- Write dev and test prediction files through the provided `test` flow.
- Prefer deterministic test dataloader behavior when changing prediction code;
  the IDs must still match the prediction rows written to disk.

## Verification Checklist

Before a full run:

- Part I GPT-2 checks still pass if the paraphrase work touched shared model
  code.
- A paraphrase batch has:
  - `token_ids`: rank 2
  - `attention_mask`: same first two dimensions as `token_ids`
  - `labels`: one binary label per training example
- A model forward call returns `(batch_size, 2)` logits.
- `F.cross_entropy(logits, labels)` succeeds on one batch.
- A small dev evaluation path reaches `argmax` and reports binary predictions.

After a full run:

- Best checkpoint exists and can be loaded by `test`.
- Dev output and test output files are created.
- Prediction values are binary and row IDs come from the dataset batches.
- Experiment result is added to the log table below.

## Experiment Queue

Start with a minimal correct cloze baseline, then change one factor at a time.

## Current Baseline Snapshot

The current cloze verbalizer run in `predictions/para-dev-output.csv` was
checked against `data/quora-dev.csv` by matching rows with `id`.

| Metric | Value |
| --- | ---: |
| Matched dev examples | 40,429 |
| Accuracy | 0.89797 |
| Macro F1 | 0.89050 |
| Positive precision | 0.85950 |
| Positive recall | 0.86429 |
| Positive F1 | 0.86189 |

Confusion matrix:

| | Pred 0 | Pred 1 |
| --- | ---: | ---: |
| Gold 0 | 23,433 | 2,104 |
| Gold 1 | 2,021 | 12,871 |

Interpretation:

- The baseline is strong enough to use as the main comparison point.
- False positives often come from pairs with overlapping topics but different
  conditions, targets, or intent.
- False negatives often come from paraphrases with changed wording, extra
  context, typos, or reordered phrases.
- The next improvements should focus on reducing topic-overlap false positives
  while preserving recall on meaning-preserving rewrites.

## Error Analysis And Visualization Plan

Use dev predictions to build a small analysis report before changing the model.
This makes the improvement work easier to justify in the final writeup.

Recommended analysis artifacts:

1. Confusion matrix heatmap.
2. Class-wise precision, recall, and F1 bar chart.
3. False-positive and false-negative sample table with `sentence1`,
   `sentence2`, gold label, and predicted label.
4. Accuracy by sentence length bucket:
   - short-short
   - short-long
   - long-short
   - long-long
5. Accuracy by token-overlap bucket, using a simple Jaccard score over
   preprocessed whitespace tokens.
6. If probability scores are saved later, confidence histograms for correct
   predictions versus wrong predictions.

Questions to answer from the plots:

- Are high-overlap negative pairs the main false-positive source?
- Are long asymmetric pairs harder than short balanced pairs?
- Does the model underperform on label `1`, label `0`, or both equally?
- Are most wrong predictions low confidence, or are there confident systematic
  errors that need data/model changes?

## Compliance Review Against Project Guide

The added improvement plan is compatible with the project guide as long as the
following boundaries are preserved:

- Use only the provided Quora train/dev splits for model fitting, threshold
  selection, prompt selection, error analysis, and visualization.
- Use the test split only for final prediction-file generation. Do not inspect
  test labels or tune decisions against test feedback.
- Do not manually edit prediction CSV rows to improve scores. All prediction
  files should be generated by the model/evaluation code.
- Keep the paraphrase task framed as cloze-style `yes`/`no` prediction, or make
  any linear-head baseline an explicit ablation rather than silently replacing
  the required cloze setup.
- Bidirectional inference, threshold tuning, prompt ensembles, swap
  augmentation, hard-negative emphasis, class weighting, gradient clipping, and
  early stopping are allowed because they use only train/dev data and preserve
  binary output labels.
- LoRA/ReFT-style PEFT ideas are mentioned in the guide as possible extensions,
  but should only be attempted after checking dependency and submission
  constraints. The safer default is to avoid adding new packages unless clearly
  necessary.
- The local `data/quora-dev.csv` currently contains 40,429 matched rows in the
  prediction check, while the guide text lists 20,215 dev examples. Treat the
  repository files as the executable source of truth for experiments, and note
  this mismatch if reporting exact counts.

## Improvement Roadmap

Prioritize low-risk changes that preserve the existing assignment contract:
binary labels, deterministic prediction rows, and no external test-label use.

### Phase 1: Inference-Time Improvements

These should be tried first because they do not require a full retraining run.

1. Bidirectional inference:
   - Evaluate both `(sentence1, sentence2)` and `(sentence2, sentence1)`.
   - Average the two `[no, yes]` logit vectors before prediction.
   - Rationale: paraphrase is symmetric, but the current prompt is directional.
   - Expected effect: reduce order-sensitive errors with minimal code change.

2. Threshold tuning:
   - Convert `[no, yes]` logits into `P(yes)` on the dev split.
   - Search thresholds such as `0.40` to `0.60` with step `0.01`.
   - Select the threshold by dev accuracy first, then macro F1 as tie-breaker.
   - Keep the selected threshold fixed for test prediction.
   - Rationale: `argmax` is equivalent to threshold `0.50`, which may not be
     optimal for the class distribution.

3. Prompt-variant ensemble:
   - Try two or three semantically equivalent prompts.
   - Average logits across prompts, keeping the same `no`/`yes` verbalizer.
   - Candidate prompts:
     - `Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no":`
     - `Do these two questions ask the same thing? Answer "yes" or "no":`
     - `Are the following questions semantically equivalent? Answer "yes" or "no":`
   - Rationale: GPT-style cloze classifiers can be sensitive to prompt wording.

### Phase 2: Training Data Improvements

These require retraining, so run them after Phase 1 establishes a stronger
inference baseline.

1. Sentence-order swap augmentation:
   - Add `(sentence2, sentence1, label)` for each training pair.
   - Keep dev/test unchanged.
   - Rationale: teaches the model that paraphrase labels are symmetric.
   - Risk: doubles training examples and therefore training time.

2. Hard-negative emphasis:
   - Identify train examples with high token overlap and label `0`.
   - Oversample them lightly or add a weighted loss variant.
   - Rationale: current false positives often look like high-overlap
     non-paraphrases.
   - Risk: too much emphasis may lower paraphrase recall.

3. Class weighting:
   - Use weighted cross entropy if dev recall/precision becomes imbalanced.
   - Start conservatively because the current positive precision and recall are
     already balanced.

### Phase 3: Training Stability

Apply one optimization change at a time and record all settings.

1. Learning-rate sweep:
   - `5e-6`
   - `1e-5`
   - `2e-5`

2. Weight decay:
   - Compare `0.0` versus `0.01`.

3. Gradient clipping:
   - Add `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` after
     `loss.backward()`.
   - Rationale: reduces unstable updates during full-model fine-tuning.

4. Early stopping:
   - Keep best-dev-accuracy checkpointing.
   - Stop if dev accuracy does not improve for a small patience window.

### Phase 4: Model Ablations

Use these for the final report if time remains.

1. Cloze verbalizer versus linear classification head:
   - Cloze: last hidden state to vocabulary logits, select `no` and `yes`.
   - Linear: last hidden state to `nn.Linear(hidden_size, 2)`.
   - Report which objective works better for this assignment setup.

2. Model size comparison:
   - `gpt2` first.
   - `gpt2-medium` only if GPU memory and time allow.
   - Avoid larger runs unless the smaller model experiments are already clean.

3. Freeze policy:
   - Full-model fine-tuning.
   - Partial freezing as a speed/memory baseline.

## Recommended Final Experiment Package

The most practical high-value package is:

1. Current cloze baseline.
2. Cloze baseline plus bidirectional inference.
3. Cloze baseline plus bidirectional inference and threshold tuning.
4. Swap-augmented retraining plus the best inference recipe.

Report each experiment with:

- dev accuracy
- dev macro F1
- confusion matrix
- changed implementation details
- one short error-analysis note

Potential experiments:

1. Cloze verbalizer logits versus the starter linear two-class head.
2. Prompt wording and question order variants while keeping split consistency.
3. Sentence-order augmentation: train on both `(s1, s2)` and `(s2, s1)` when
   the label is symmetric.
4. Full-model fine-tuning versus selectively frozen GPT-2 parameters.
5. `gpt2`, `gpt2-medium`, and `gpt2-large` subject to GPU memory limits.
6. Learning rate, batch size, gradient stability, and early stopping behavior.
7. Parameter-efficient fine-tuning ideas mentioned by the PDF, such as LoRA or
   ReFT, only after the baseline is correct and dependency constraints are
   checked.

## Experiment Log

| Date | Model | Prompt / Verbalizer | Train Change | Dev Acc | Dev F1 | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| | | | | | | |

## Working Rules For Future Codex Changes

- Read this file and the current paraphrase code before editing.
- Keep changes scoped to Paraphrase Detection unless a shared GPT-2 fix is
  required.
- Do not assume the starter paraphrase data path is internally consistent; check
  prompt, label, and model-output contracts together.
- Preserve the provided train/dev/test ethics boundary.
- Make the smallest runnable baseline first, verify tensor shapes, then iterate
  on modeling improvements with recorded dev results.

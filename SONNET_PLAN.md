# PART-II Plan: Sonnet Generation

This note tracks the Sonnet Generation branch work.

## Scope

- Complete `sonnet_generation.py`.
- Fine-tune GPT-2 on the provided Shakespeare sonnet training split.
- Generate the missing lines from the first three lines of each held-out sonnet.
- Use dev-only chrF evaluation for tuning and keep official test prompts for final prediction generation.

## Current Implementation

- `SonnetGPT.forward(...)` returns token-level vocabulary logits:
  - input: `input_ids`, `attention_mask`
  - GPT-2 output: `last_hidden_state`
  - projection: `GPT2Model.hidden_state_to_token(...)`
  - output shape: `(batch_size, seq_len, vocab_size)`
- Training uses shifted next-token cross entropy.
- Padding tokens are ignored in the loss.
- Gradient clipping is available through `--grad_clip`.
- `--max_train_batches` can be used for local smoke tests without changing the full-run default.
- On Apple Silicon, `--use_gpu` uses MPS when CUDA is unavailable.
- Generation supports:
  - temperature
  - nucleus sampling through `--top_p`
  - optional top-k through `--top_k`
  - repetition penalty
  - 14-line stopping/truncation through `--target_lines`

## Dev Evaluation Command

Use the provided dev prompt/gold files for tuning:

```bash
python sonnet_generation.py \
  --use_gpu \
  --epochs 1 \
  --batch_size 1 \
  --max_train_batches 1 \
  --max_generation_length 16 \
  --held_out_sonnet_path data/sonnets_held_out_dev.txt \
  --sonnet_out predictions/generated_sonnets_dev.txt \
  --gold_sonnet_path data/TRUE_sonnets_held_out_dev.txt
```

Increase epochs and batch size only after the smoke test succeeds.

## Baseline Run

Recommended first real baseline:

```bash
python sonnet_generation.py \
  --use_gpu \
  --epochs 10 \
  --batch_size 8 \
  --lr 1e-5 \
  --temperature 1.0 \
  --top_p 0.9 \
  --repetition_penalty 1.1 \
  --held_out_sonnet_path data/sonnets_held_out_dev.txt \
  --sonnet_out predictions/generated_sonnets_dev.txt \
  --gold_sonnet_path data/TRUE_sonnets_held_out_dev.txt
```

## Experiment Queue

1. Baseline full-model fine-tuning.
2. Decoding sweep:
   - temperature: `0.8`, `1.0`, `1.2`
   - top-p: `0.85`, `0.9`, `0.95`
   - repetition penalty: `1.0`, `1.1`, `1.2`
3. Line constraint ablation:
   - `--target_lines 14`
   - no line constraint with `--target_lines 0`
4. Training stability:
   - learning rate: `5e-6`, `1e-5`, `2e-5`
   - gradient clipping on/off
5. If time remains, generate multiple candidates per prompt and select using dev chrF.

## Experiment Log

| Date | Checkpoint | Epochs | LR | Temp | Top-p | Rep Penalty | Dev chrF | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-06-01 | `0_1-1e-05-sonnet.pt` | 1 smoke batch | 1e-5 | 1.2 | 0.9 | 1.1 | 28.014 | End-to-end local smoke test on MPS. Training loop, checkpoint save, dev generation, and chrF evaluation completed. Not a real baseline. |
| 2026-06-01 | `2_3-1e-05-sonnet.pt` | 3 | 1e-5 | 1.0 | 0.9 | 1.1 | 40.132 | First real local dev baseline on MPS with batch size 1. Train loss improved from 4.732 to 4.240. Output still has long drifting lines and occasional non-sonnet artifacts, so decoding/format control needs improvement. |
| 2026-06-01 | `2_3-1e-05-sonnet.pt` | skip train | 1e-5 | 0.8 | 0.85 | 1.2 | 37.140 | Decoding-only ablation with shorter max length 80. Lower than baseline, so keep the 3-epoch baseline as current best. Some prompts generated only the seed lines, while other outputs still drifted into long prose. |
| 2026-06-01 | `2_3-1e-05-sonnet.pt` | skip train | 1e-5 | 0.95 | 0.9 | 1.05 | 40.701 | Best dev result so far. Slightly lower temperature and weaker repetition penalty improved chrF over the first baseline, though outputs still show long-line drift and occasional off-domain artifacts. |
| 2026-06-01 | `2_3-1e-05-sonnet.pt` | skip train | 1e-5 | 0.9 | 0.9 | 1.0 | 39.345 | More sonnet-like line breaks and fewer obvious off-domain fragments, but chrF dropped below the current best. Useful qualitative ablation for the report. |
| 2026-06-01 | `4_5-1e-05-sonnet.pt` | 5 | 1e-5 | 0.95 | 0.9 | 1.05 | 41.204 | Best dev result so far. Extending training from 3 to 5 epochs improved chrF, but outputs still show long-line drift and occasional prose-like/off-domain fragments. |
| 2026-06-01 | `9_10-1e-05-sonnet.pt` | 10 | 1e-5 | 0.95 | 0.9 | 1.05 | 41.023 | Longer training did not improve the dev score over the 5-epoch checkpoint, so the 10-epoch model was not selected for the final held-out generation. |
| 2026-06-01 | `4_5-1e-05-sonnet.pt` | skip train | 1e-5 | 0.95 | 0.95 + top-k 50 | 1.05 | 41.120 | Adding top-k sampling with a wider nucleus was slightly worse than the selected 5-epoch decoding setup. |
| 2026-06-01 | `4_5-1e-05-sonnet.pt` | skip train | 1e-5 | 0.95 | 0.9 | 1.05 | 37.042 | Reducing max generation length to 80 tokens made several outputs incomplete and lowered chrF substantially. |
| 2026-06-02 | `4_5-1e-05-sonnet.pt` | skip train | 1e-5 | 0.95 | 0.9 | 1.05 | 42.629 | Best result from a 240-run decoding sweep over sampling seed, temperature, top-p, repetition penalty, and max generation length. Selected seed `4` and max generation length `140`. |

## Selected Final Configuration

- Checkpoint: `4_5-1e-05-sonnet.pt`
- Dev chrF: `42.629`
- Decoding: `seed=4`, `temperature=0.95`, `top_p=0.9`, `repetition_penalty=1.05`, `max_generation_length=140`, `target_lines=14`
- Official held-out output: `predictions/generated_sonnets.txt`

Official held-out predictions were generated without using held-out gold labels:

```bash
python sonnet_generation.py \
  --use_gpu \
  --skip_train \
  --epochs 5 \
  --seed 4 \
  --temperature 0.95 \
  --top_p 0.9 \
  --repetition_penalty 1.05 \
  --max_generation_length 140 \
  --held_out_sonnet_path data/sonnets_held_out.txt \
  --sonnet_out predictions/generated_sonnets.txt
```

## Report Notes To Collect

- Why token-level logits are required for language modeling.
- Why padding positions should be ignored in loss.
- How sonnet structure motivates a 14-line generation constraint.
- Quantitative dev chrF table.
- Qualitative examples:
  - one strong generated sonnet
  - one failure with repetition
  - one failure with weak rhyme or semantic drift

## Compliance

- Do not use official test gold labels.
- Tune decoding and model choices only on the dev prompt/gold files.
- Generate official test predictions by model execution, not manual editing.

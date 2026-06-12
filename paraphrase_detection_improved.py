#!/usr/bin/env python3
"""
Improved GPT-2 cloze-style paraphrase detection.

This file intentionally leaves paraphrase_detection.py unchanged. It keeps the
same GPT-2 backbone and yes/no cloze objective, but adds fine-tuning choices
motivated by the error analysis report:

- high-overlap negative weighting
- low-overlap positive weighting
- optional sentence-order swap augmentation
- optional bidirectional training loss
- optional bidirectional/prompt-ensemble evaluation
- optional dev-threshold calibration
"""

import argparse
import csv
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import GPT2Tokenizer

from datasets import (
  PARAPHRASE_PROMPT_TEMPLATES,
  augment_with_sentence_swaps,
  format_paraphrase_prompt,
  get_prompt_template_names,
  lexical_jaccard,
  load_paraphrase_data,
)
from evaluation import model_eval_paraphrase, model_test_paraphrase, tune_paraphrase_threshold
from optimizer import AdamW
from paraphrase_detection import (
  ParaphraseGPT,
  add_arguments,
  build_checkpoint_eval_args as build_baseline_checkpoint_eval_args,
  compute_class_weights,
  maybe_data_parallel,
  resolve_device,
  save_model,
  seed_everything,
)

TQDM_DISABLE = False

CHECKPOINT_RUNTIME_ARGS = (
  "para_dev",
  "para_test",
  "para_dev_out",
  "para_test_out",
  "filepath",
  "batch_size",
  "use_gpu",
  "gpu_ids",
  "multi_gpu",
)

CHECKPOINT_EVAL_ARGS = (
  "bidirectional_eval",
  "prompt_template",
  "prompt_ensemble_eval",
  "prompt_ensemble_templates",
  "threshold",
  "tune_threshold",
  "threshold_start",
  "threshold_stop",
  "threshold_step",
)


class ImprovedParaphraseDetectionDataset(Dataset):
  def __init__(self, dataset, args, split="train"):
    self.dataset = dataset
    self.p = args
    self.split = split
    self.is_test = split == "test"
    self.is_train = split == "train"
    self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.train_prompt_names = self._parse_prompt_names(getattr(args, "train_prompt_templates", ""))

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.dataset[idx]

  def _parse_prompt_names(self, raw_names):
    names = [name.strip() for name in str(raw_names).split(",") if name.strip()]
    unknown = [name for name in names if name not in PARAPHRASE_PROMPT_TEMPLATES]
    if unknown:
      raise ValueError(f"Unknown prompt template(s): {', '.join(unknown)}")
    return names

  def _prompt_names_for_batch(self, batch_size):
    if self.is_train and self.train_prompt_names:
      if getattr(self.p, "random_train_prompt", False):
        return [random.choice(self.train_prompt_names) for _ in range(batch_size)]
      return [self.train_prompt_names[i % len(self.train_prompt_names)] for i in range(batch_size)]
    return [get_prompt_template_names(self.p)[0] for _ in range(batch_size)]

  def _encode_pairs(self, sent1, sent2, prompt_names):
    prompts = [
      format_paraphrase_prompt(s1, s2, prompt_name)
      for s1, s2, prompt_name in zip(sent1, sent2, prompt_names)
    ]
    encoding = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    return torch.LongTensor(encoding["input_ids"]), torch.LongTensor(encoding["attention_mask"])

  def _example_weights(self, sent1, sent2, labels):
    weights = torch.ones(len(sent1), dtype=torch.float)
    hard_negative_weight = getattr(self.p, "hard_negative_weight", 1.0)
    hard_negative_jaccard = getattr(self.p, "hard_negative_jaccard", 0.6)
    low_positive_weight = getattr(self.p, "low_overlap_positive_weight", 1.0)
    low_positive_jaccard = getattr(self.p, "low_overlap_positive_jaccard", 0.2)

    for i, (s1, s2, label) in enumerate(zip(sent1, sent2, labels)):
      overlap = lexical_jaccard(s1, s2)
      label = int(label.item())
      if label == 0 and hard_negative_weight > 1.0 and overlap >= hard_negative_jaccard:
        weights[i] = hard_negative_weight
      elif label == 1 and low_positive_weight > 1.0 and overlap <= low_positive_jaccard:
        weights[i] = low_positive_weight
    return weights

  def collate_fn(self, all_data):
    sent1 = [x[0] for x in all_data]
    sent2 = [x[1] for x in all_data]
    sent_ids = [x[2] if self.is_test else x[3] for x in all_data]
    prompt_names = self._prompt_names_for_batch(len(all_data))
    token_ids, attention_mask = self._encode_pairs(sent1, sent2, prompt_names)

    batched_data = {
      "token_ids": token_ids,
      "attention_mask": attention_mask,
      "sent_ids": sent_ids,
    }

    if not self.is_test:
      labels = torch.LongTensor([x[2] for x in all_data])
      batched_data["labels"] = labels
      batched_data["example_weights"] = self._example_weights(sent1, sent2, labels)

    needs_reversed = (
      (getattr(self.p, "bidirectional_train", False) and self.is_train)
      or getattr(self.p, "bidirectional_eval", False)
      or getattr(self.p, "tune_threshold", False)
    )
    if needs_reversed:
      reversed_ids, reversed_mask = self._encode_pairs(sent2, sent1, prompt_names)
      batched_data["reversed_token_ids"] = reversed_ids
      batched_data["reversed_attention_mask"] = reversed_mask

    if not self.is_train and getattr(self.p, "prompt_ensemble_eval", False):
      ensemble_token_ids = []
      ensemble_attention_mask = []
      reversed_ensemble_token_ids = []
      reversed_ensemble_attention_mask = []
      for prompt_name in get_prompt_template_names(self.p, ensemble=True):
        prompt_batch = [prompt_name] * len(all_data)
        ids, mask = self._encode_pairs(sent1, sent2, prompt_batch)
        ensemble_token_ids.append(ids)
        ensemble_attention_mask.append(mask)
        if getattr(self.p, "bidirectional_eval", False) or getattr(self.p, "tune_threshold", False):
          r_ids, r_mask = self._encode_pairs(sent2, sent1, prompt_batch)
          reversed_ensemble_token_ids.append(r_ids)
          reversed_ensemble_attention_mask.append(r_mask)

      batched_data["ensemble_token_ids"] = ensemble_token_ids
      batched_data["ensemble_attention_mask"] = ensemble_attention_mask
      if reversed_ensemble_token_ids:
        batched_data["reversed_ensemble_token_ids"] = reversed_ensemble_token_ids
        batched_data["reversed_ensemble_attention_mask"] = reversed_ensemble_attention_mask

    return batched_data


def build_checkpoint_eval_args(checkpoint_args, cli_args):
  eval_args = build_baseline_checkpoint_eval_args(checkpoint_args, cli_args)

  for name in CHECKPOINT_RUNTIME_ARGS:
    if hasattr(cli_args, name):
      setattr(eval_args, name, getattr(cli_args, name))

  for name in CHECKPOINT_EVAL_ARGS:
    if not hasattr(eval_args, name) and hasattr(cli_args, name):
      setattr(eval_args, name, getattr(cli_args, name))

  if getattr(cli_args, "override_checkpoint_eval_args", False):
    for name in CHECKPOINT_EVAL_ARGS:
      if hasattr(cli_args, name):
        setattr(eval_args, name, getattr(cli_args, name))

  return eval_args


def build_checkpoint_path(args):
  parts = ["improved", args.model_size, str(args.epochs), str(args.lr)]
  if getattr(args, "hard_negative_weight", 1.0) > 1.0:
    parts.append("hardneg")
  if getattr(args, "low_overlap_positive_weight", 1.0) > 1.0:
    parts.append("lowpos")
  if getattr(args, "augment_swap", False):
    parts.append("swap")
  if getattr(args, "bidirectional_train", False):
    parts.append("bidirtrain")
  if getattr(args, "consistency_weight", 0.0) > 0.0:
    parts.append("consistency")
  if getattr(args, "random_train_prompt", False):
    parts.append("promptmix")
  if getattr(args, "classification_head", False):
    parts.append("linear")
  return "-".join(parts + ["paraphrase.pt"])


def _weighted_mean(values, weights):
  return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def _cross_entropy_loss(logits, labels, example_weights, class_weights=None):
  per_example_loss = F.cross_entropy(logits, labels, weight=class_weights, reduction="none")
  return _weighted_mean(per_example_loss, example_weights)


def _symmetric_kl_loss(logits_a, logits_b, example_weights):
  probs_a = F.softmax(logits_a.detach(), dim=-1)
  probs_b = F.softmax(logits_b.detach(), dim=-1)
  log_probs_a = F.log_softmax(logits_a, dim=-1)
  log_probs_b = F.log_softmax(logits_b, dim=-1)
  kl_ab = F.kl_div(log_probs_a, probs_b, reduction="none").sum(dim=-1)
  kl_ba = F.kl_div(log_probs_b, probs_a, reduction="none").sum(dim=-1)
  return _weighted_mean(0.5 * (kl_ab + kl_ba), example_weights)


def _set_scheduled_lr(optimizer, base_lr, global_step, total_steps, warmup_steps, lr_decay):
  if total_steps <= 0 or (warmup_steps <= 0 and not lr_decay):
    return

  step = global_step + 1
  if warmup_steps > 0 and step <= warmup_steps:
    scale = step / float(warmup_steps)
  elif lr_decay:
    remaining = max(total_steps - step, 0)
    decay_steps = max(total_steps - warmup_steps, 1)
    scale = remaining / float(decay_steps)
  else:
    scale = 1.0

  for group in optimizer.param_groups:
    group["lr"] = base_lr * max(scale, 0.0)


def train(args):
  device = resolve_device(args)
  args = add_arguments(args)
  para_train_data = load_paraphrase_data(args.para_train)
  para_dev_data = load_paraphrase_data(args.para_dev)

  if args.augment_swap:
    para_train_data = augment_with_sentence_swaps(para_train_data)
    print(f"Applied sentence-order swap augmentation: {len(para_train_data)} train examples")

  class_weights = None
  if args.class_weighting:
    class_weights = compute_class_weights(para_train_data).to(device)
    print(f"Using class weights: no={class_weights[0].item():.3f}, yes={class_weights[1].item():.3f}")

  train_dataset = ImprovedParaphraseDetectionDataset(para_train_data, args, split="train")
  dev_dataset = ImprovedParaphraseDetectionDataset(para_dev_data, args, split="dev")
  train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=args.batch_size,
    collate_fn=train_dataset.collate_fn,
  )
  dev_dataloader = DataLoader(
    dev_dataset,
    shuffle=False,
    batch_size=args.batch_size,
    collate_fn=dev_dataset.collate_fn,
  )

  model = ParaphraseGPT(args).to(device)
  model = maybe_data_parallel(model, args)
  optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

  total_steps = len(train_dataloader) * args.epochs
  warmup_steps = int(total_steps * args.warmup_ratio)
  global_step = 0
  best_dev_acc = 0.0
  best_dev_f1 = 0.0
  epochs_without_improvement = 0

  for epoch in range(args.epochs):
    model.train()
    train_loss = 0.0
    num_batches = 0

    for batch in tqdm(train_dataloader, desc=f"improved-train-{epoch}", disable=TQDM_DISABLE):
      labels = batch["labels"].flatten().to(device)
      example_weights = batch["example_weights"].to(device)
      b_ids = batch["token_ids"].to(device)
      b_mask = batch["attention_mask"].to(device)

      _set_scheduled_lr(optimizer, args.lr, global_step, total_steps, warmup_steps, args.lr_decay)
      optimizer.zero_grad()

      logits = model(b_ids, b_mask)
      loss = _cross_entropy_loss(logits, labels, example_weights, class_weights=class_weights)

      if args.bidirectional_train:
        rb_ids = batch["reversed_token_ids"].to(device)
        rb_mask = batch["reversed_attention_mask"].to(device)
        reversed_logits = model(rb_ids, rb_mask)
        reversed_loss = _cross_entropy_loss(
          reversed_logits,
          labels,
          example_weights,
          class_weights=class_weights,
        )
        loss = 0.5 * (loss + reversed_loss)
        if args.consistency_weight > 0:
          loss = loss + args.consistency_weight * _symmetric_kl_loss(logits, reversed_logits, example_weights)

      loss.backward()
      if args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1
      global_step += 1

    train_loss = train_loss / max(num_batches, 1)
    dev_acc, dev_f1, *_ = model_eval_paraphrase(
      dev_dataloader,
      model,
      device,
      bidirectional=args.bidirectional_eval,
    )

    improved = dev_acc > best_dev_acc or (dev_acc == best_dev_acc and dev_f1 > best_dev_f1)
    if improved:
      best_dev_acc = dev_acc
      best_dev_f1 = dev_f1
      epochs_without_improvement = 0
      save_model(model, optimizer, args, args.filepath)
    else:
      epochs_without_improvement += 1

    print(
      f"Epoch {epoch}: train loss :: {train_loss :.3f}, "
      f"dev acc :: {dev_acc :.3f}, dev macro-F1 :: {dev_f1 :.3f}"
    )
    if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
      print(f"Early stopping after {epochs_without_improvement} epochs without dev improvement")
      break


@torch.no_grad()
def test(args):
  device = resolve_device(args)
  saved = torch.load(args.filepath, map_location=device, weights_only=False)
  model_args = saved["args"]
  if not all(hasattr(model_args, name) for name in ("d", "l", "num_heads")):
    model_args = add_arguments(model_args)
  eval_args = build_checkpoint_eval_args(model_args, args)

  model = ParaphraseGPT(model_args)
  model.load_state_dict(saved["model"])
  model = model.to(device)
  model = maybe_data_parallel(model, args)
  model.eval()
  print(f"Loaded improved model to test from {args.filepath}")

  para_dev_data = load_paraphrase_data(eval_args.para_dev)
  para_test_data = load_paraphrase_data(eval_args.para_test, split="test")
  para_dev_dataset = ImprovedParaphraseDetectionDataset(para_dev_data, eval_args, split="dev")
  para_test_dataset = ImprovedParaphraseDetectionDataset(para_test_data, eval_args, split="test")
  para_dev_dataloader = DataLoader(
    para_dev_dataset,
    shuffle=False,
    batch_size=eval_args.batch_size,
    collate_fn=para_dev_dataset.collate_fn,
  )
  para_test_dataloader = DataLoader(
    para_test_dataset,
    shuffle=False,
    batch_size=eval_args.batch_size,
    collate_fn=para_test_dataset.collate_fn,
  )

  threshold = getattr(eval_args, "threshold", None)
  bidirectional_eval = getattr(eval_args, "bidirectional_eval", False)
  if getattr(eval_args, "tune_threshold", False):
    threshold, tuned_acc, tuned_f1 = tune_paraphrase_threshold(
      para_dev_dataloader,
      model,
      device,
      bidirectional=bidirectional_eval,
      start=getattr(eval_args, "threshold_start", 0.35),
      stop=getattr(eval_args, "threshold_stop", 0.70),
      step=getattr(eval_args, "threshold_step", 0.01),
    )
    print(f"best dev threshold :: {threshold :.2f}, acc :: {tuned_acc :.3f}, f1 :: {tuned_f1 :.3f}")

  dev_acc, dev_f1, dev_pred, _, dev_ids = model_eval_paraphrase(
    para_dev_dataloader,
    model,
    device,
    bidirectional=bidirectional_eval,
    threshold=threshold,
  )
  print(f"dev paraphrase acc :: {dev_acc :.3f}, macro-F1 :: {dev_f1 :.3f}")

  test_pred, test_ids = model_test_paraphrase(
    para_test_dataloader,
    model,
    device,
    bidirectional=bidirectional_eval,
    threshold=threshold,
  )

  with open(eval_args.para_dev_out, "w+", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "Predicted_Is_Paraphrase"])
    for sent_id, pred in zip(dev_ids, dev_pred):
      writer.writerow([sent_id, pred])

  with open(eval_args.para_test_out, "w+", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "Predicted_Is_Paraphrase"])
    for sent_id, pred in zip(test_ids, test_pred):
      writer.writerow([sent_id, pred])

  return {
    "dev_accuracy": dev_acc,
    "dev_macro_f1": dev_f1,
    "threshold": threshold,
    "dev_output": eval_args.para_dev_out,
    "test_output": eval_args.para_test_out,
  }


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
  parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
  parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")
  parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-improved-output.csv")
  parser.add_argument("--para_test_out", type=str, default="predictions/para-test-improved-output.csv")
  parser.add_argument("--filepath", type=str, default=None)

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action="store_true")
  parser.add_argument("--multi_gpu", action="store_true")
  parser.add_argument("--gpu_ids", type=str, default=None)
  parser.add_argument("--skip_train", action="store_true")

  parser.add_argument("--prompt_template", type=str, default="paraphrase", choices=list(PARAPHRASE_PROMPT_TEMPLATES))
  parser.add_argument("--train_prompt_templates", type=str, default="paraphrase")
  parser.add_argument("--random_train_prompt", action="store_true")
  parser.add_argument("--prompt_ensemble_eval", action="store_true")
  parser.add_argument("--prompt_ensemble_templates", type=str, default="paraphrase,duplicate,equivalent")
  parser.add_argument("--bidirectional_eval", action="store_true")
  parser.add_argument("--bidirectional_train", action="store_true")
  parser.add_argument("--consistency_weight", type=float, default=0.0)

  parser.add_argument("--threshold", type=float, default=None)
  parser.add_argument("--tune_threshold", action="store_true")
  parser.add_argument("--threshold_start", type=float, default=0.35)
  parser.add_argument("--threshold_stop", type=float, default=0.70)
  parser.add_argument("--threshold_step", type=float, default=0.01)
  parser.add_argument("--override_checkpoint_eval_args", action="store_true")

  parser.add_argument("--augment_swap", action="store_true")
  parser.add_argument("--hard_negative_weight", type=float, default=1.0)
  parser.add_argument("--hard_negative_jaccard", type=float, default=0.6)
  parser.add_argument("--low_overlap_positive_weight", type=float, default=1.0)
  parser.add_argument("--low_overlap_positive_jaccard", type=float, default=0.2)
  parser.add_argument("--class_weighting", action="store_true")
  parser.add_argument("--classification_head", action="store_true")
  parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)

  parser.add_argument("--grad_clip", type=float, default=1.0)
  parser.add_argument("--weight_decay", type=float, default=0.01)
  parser.add_argument("--warmup_ratio", type=float, default=0.06)
  parser.add_argument("--lr_decay", action="store_true")
  parser.add_argument("--early_stopping_patience", type=int, default=2)

  parser.add_argument("--batch_size", type=int, default=128)
  parser.add_argument("--lr", type=float, default=1e-5)
  parser.add_argument("--model_size", type=str, choices=["gpt2", "gpt2-medium", "gpt2-large"], default="gpt2")

  args = parser.parse_args()
  args.filepath = args.filepath or build_checkpoint_path(args)
  return args


if __name__ == "__main__":
  args = get_args()
  seed_everything(args.seed)
  if not args.skip_train:
    train(args)
  test(args)

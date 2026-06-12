#!/usr/bin/env python3
"""
No-train decision policy search for paraphrase detection checkpoints.

This script keeps the trained baseline and improved checkpoints fixed, then
searches inference-time policies on the dev set:

- threshold tuning for each model
- improved-as-veto cascade
- improved only for uncertain baseline scores
- segment-gated improved calls
- lexical post-processing veto rules
- weighted score ensembles
- cost/call-rate-aware policy selection

The selected policy is then applied to dev and test predictions.
"""

import argparse
import csv
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import (
  ParaphraseDetectionDataset,
  ParaphraseDetectionTestDataset,
  lexical_jaccard,
  load_paraphrase_data,
)
from evaluation import _batch_paraphrase_logits
from paraphrase_detection import (
  ParaphraseGPT,
  add_arguments,
  build_checkpoint_eval_args as build_base_eval_args,
  maybe_data_parallel,
  resolve_device,
)
from paraphrase_detection_improved import (
  ImprovedParaphraseDetectionDataset,
  build_checkpoint_eval_args as build_improved_eval_args,
)


DEFAULT_IMPROVED_CHECKPOINT = (
  "improved-gpt2-10-1e-05-hardneg-lowpos-bidirtrain-consistency-promptmix-paraphrase.pt"
)

NEGATION_WORDS = {
  "not",
  "no",
  "never",
  "none",
  "without",
  "cannot",
  "can't",
  "dont",
  "don't",
  "doesnt",
  "doesn't",
  "isnt",
  "isn't",
}

OPPOSITION_PAIRS = (
  ("best", "worst"),
  ("better", "worse"),
  ("good", "bad"),
  ("more", "less"),
  ("most", "least"),
  ("increase", "decrease"),
  ("increased", "decreased"),
  ("increases", "decreases"),
  ("increasing", "decreasing"),
  ("higher", "lower"),
  ("highest", "lowest"),
  ("large", "small"),
  ("larger", "smaller"),
  ("largest", "smallest"),
  ("long", "short"),
  ("longer", "shorter"),
  ("longest", "shortest"),
  ("first", "last"),
  ("before", "after"),
  ("with", "without"),
  ("true", "false"),
  ("yes", "no"),
  ("male", "female"),
  ("man", "woman"),
  ("men", "women"),
  ("boy", "girl"),
  ("boys", "girls"),
)


def frange(start, stop, step):
  values = []
  x = start
  while x <= stop + step / 2:
    values.append(round(float(x), 6))
    x += step
  return values


def softmax_yes_prob(logits):
  logits = logits - logits.max(axis=1, keepdims=True)
  exp_logits = np.exp(logits)
  return exp_logits[:, 1] / exp_logits.sum(axis=1)


def rows_to_ids_and_labels(rows, split):
  if split == "test":
    ids = np.asarray([row[2] for row in rows])
    labels = None
  else:
    ids = np.asarray([row[3] for row in rows])
    labels = np.asarray([row[2] for row in rows], dtype=np.int64)
  return ids, labels


def token_set(sentence):
  return set(sentence.split())


def has_opposition(sent1, sent2):
  tokens1 = token_set(sent1)
  tokens2 = token_set(sent2)

  if bool(tokens1 & NEGATION_WORDS) != bool(tokens2 & NEGATION_WORDS):
    return True

  for left, right in OPPOSITION_PAIRS:
    if (left in tokens1 and right in tokens2) or (right in tokens1 and left in tokens2):
      return True

  numbers1 = set(re.findall(r"\d+", sent1))
  numbers2 = set(re.findall(r"\d+", sent2))
  return bool(numbers1 and numbers2 and numbers1 != numbers2)


def build_features(rows, split):
  sent1 = [row[0] for row in rows]
  sent2 = [row[1] for row in rows]
  jaccard = np.asarray([lexical_jaccard(s1, s2) for s1, s2 in zip(sent1, sent2)], dtype=np.float32)
  len1 = np.asarray([len(s.split()) for s in sent1], dtype=np.float32)
  len2 = np.asarray([len(s.split()) for s in sent2], dtype=np.float32)
  max_len = np.maximum(len1, len2)
  min_len = np.minimum(len1, len2)
  abs_len_diff = np.abs(len1 - len2)
  suspicious = np.asarray([has_opposition(s1, s2) for s1, s2 in zip(sent1, sent2)], dtype=bool)
  ids, labels = rows_to_ids_and_labels(rows, split)
  return {
    "ids": ids,
    "labels": labels,
    "jaccard": jaccard,
    "len1": len1,
    "len2": len2,
    "max_len": max_len,
    "min_len": min_len,
    "abs_len_diff": abs_len_diff,
    "suspicious": suspicious,
  }


def cache_path(cache_dir, model_name, split):
  return Path(cache_dir) / f"{model_name}-{split}-scores.npz"


def try_load_scores(path, expected_ids, refresh_cache=False):
  if refresh_cache or not path.exists():
    return None

  cached = np.load(path, allow_pickle=True)
  ids = cached["ids"]
  if len(ids) != len(expected_ids) or np.any(ids != expected_ids):
    return None
  print(f"Loaded cached scores from {path}")
  return {
    "ids": ids,
    "labels": cached["labels"] if "labels" in cached.files else None,
    "probs": cached["probs"],
  }


def save_scores(path, ids, labels, probs):
  path.parent.mkdir(parents=True, exist_ok=True)
  if labels is None:
    np.savez_compressed(path, ids=ids, probs=probs)
  else:
    np.savez_compressed(path, ids=ids, labels=labels, probs=probs)


def score_split(model, device, rows, split, dataset_cls, eval_args, model_name):
  if split == "test":
    if dataset_cls is ImprovedParaphraseDetectionDataset:
      dataset = dataset_cls(rows, eval_args, split="test")
    else:
      dataset = ParaphraseDetectionTestDataset(rows, eval_args)
  else:
    if dataset_cls is ImprovedParaphraseDetectionDataset:
      dataset = dataset_cls(rows, eval_args, split="dev")
    else:
      dataset = ParaphraseDetectionDataset(rows, eval_args)

  dataloader = DataLoader(
    dataset,
    shuffle=False,
    batch_size=eval_args.batch_size,
    collate_fn=dataset.collate_fn,
  )

  model.eval()
  bidirectional = getattr(eval_args, "bidirectional_eval", False)
  ids, labels, probs = [], [], []
  desc = f"{model_name}-{split}-scores"
  for batch in tqdm(dataloader, desc=desc):
    with torch.no_grad():
      logits = _batch_paraphrase_logits(batch, model, device, bidirectional=bidirectional)
    probs.append(softmax_yes_prob(logits))
    ids.extend(batch["sent_ids"])
    if "labels" in batch:
      labels.extend(batch["labels"].flatten().cpu().numpy())

  ids = np.asarray(ids)
  probs = np.concatenate(probs, axis=0)
  labels = np.asarray(labels, dtype=np.int64) if labels else None
  return {"ids": ids, "labels": labels, "probs": probs}


def prepare_checkpoint_args(saved_args, args, build_eval_args):
  model_args = saved_args
  if not hasattr(model_args, "model_size"):
    model_args.model_size = "gpt2"
  if not all(hasattr(model_args, name) for name in ("d", "l", "num_heads")):
    model_args = add_arguments(model_args)
  eval_args = build_eval_args(model_args, args)
  eval_args.batch_size = args.batch_size
  return model_args, eval_args


def score_checkpoint(model_name, filepath, rows_by_split, features_by_split, args, dataset_cls, build_eval_args):
  scores = {}
  for split, rows in rows_by_split.items():
    expected_ids = features_by_split[split]["ids"]
    path = cache_path(args.cache_dir, model_name, split)
    cached = try_load_scores(path, expected_ids, refresh_cache=args.refresh_cache)
    if cached is not None:
      scores[split] = cached

  missing_splits = [split for split in rows_by_split if split not in scores]
  if not missing_splits:
    return scores

  device = resolve_device(args)
  saved = torch.load(filepath, map_location=device, weights_only=False)
  model_args, eval_args = prepare_checkpoint_args(saved["args"], args, build_eval_args)
  model = ParaphraseGPT(model_args)
  model.load_state_dict(saved["model"])
  model = model.to(device)
  model = maybe_data_parallel(model, args)
  print(f"Loaded {model_name} checkpoint from {filepath}")

  for split in missing_splits:
    split_scores = score_split(model, device, rows_by_split[split], split, dataset_cls, eval_args, model_name)
    expected_ids = features_by_split[split]["ids"]
    if len(split_scores["ids"]) != len(expected_ids) or np.any(split_scores["ids"] != expected_ids):
      raise ValueError(f"{model_name} {split} score ids do not match data order")
    save_scores(
      cache_path(args.cache_dir, model_name, split),
      split_scores["ids"],
      split_scores["labels"],
      split_scores["probs"],
    )
    scores[split] = split_scores

  del model
  if torch.cuda.is_available():
    torch.cuda.empty_cache()
  return scores


def apply_rule_veto(pred, features, rule_jaccard):
  if rule_jaccard is None:
    return pred
  pred = pred.copy()
  rule_mask = pred.astype(bool) & features["suspicious"] & (features["jaccard"] >= rule_jaccard)
  pred[rule_mask] = False
  return pred


def apply_policy(policy, base_score, improved_score, features):
  kind = policy["kind"]
  params = policy["params"]
  n = len(base_score)
  improved_called = np.zeros(n, dtype=bool)

  if kind == "base_threshold":
    pred = base_score >= params["threshold"]

  elif kind == "improved_threshold":
    improved_called[:] = True
    pred = improved_score >= params["threshold"]

  elif kind == "weighted_ensemble":
    improved_called[:] = True
    alpha = params["base_weight"]
    score = alpha * base_score + (1.0 - alpha) * improved_score
    pred = score >= params["threshold"]

  elif kind == "improved_veto":
    base_pred = base_score >= params["base_threshold"]
    improved_called = base_pred & (base_score < params["base_keep_threshold"])
    pred = base_pred.copy()
    pred[improved_called & (improved_score < params["improved_threshold"])] = False

  elif kind == "uncertain_improved":
    pred = base_score >= params["base_threshold"]
    improved_called = (base_score >= params["lower"]) & (base_score <= params["upper"])
    pred[improved_called] = improved_score[improved_called] >= params["improved_threshold"]

  elif kind == "uncertain_weighted":
    pred = base_score >= params["base_threshold"]
    improved_called = (base_score >= params["lower"]) & (base_score <= params["upper"])
    alpha = params["base_weight"]
    mixed = alpha * base_score[improved_called] + (1.0 - alpha) * improved_score[improved_called]
    pred[improved_called] = mixed >= params["threshold"]

  elif kind == "segment_veto":
    base_pred = base_score >= params["base_threshold"]
    segment = features["jaccard"] >= params["jaccard_high"]
    improved_called = base_pred & segment & (base_score < params["base_keep_threshold"])
    pred = base_pred.copy()
    pred[improved_called & (improved_score < params["improved_threshold"])] = False

  elif kind == "segment_veto_rescue":
    base_pred = base_score >= params["base_threshold"]
    high_segment = features["jaccard"] >= params["jaccard_high"]
    low_segment = features["jaccard"] <= params["jaccard_low"]
    veto_called = base_pred & high_segment & (base_score < params["base_keep_threshold"])
    rescue_called = (~base_pred) & low_segment
    improved_called = veto_called | rescue_called
    pred = base_pred.copy()
    pred[veto_called & (improved_score < params["improved_threshold"])] = False
    pred[rescue_called & (improved_score >= params["rescue_threshold"])] = True

  elif kind == "rule_veto":
    pred = base_score >= params["base_threshold"]

  else:
    raise ValueError(f"Unknown policy kind: {kind}")

  pred = apply_rule_veto(pred, features, params.get("rule_jaccard"))
  return pred.astype(np.int64), improved_called


def base_call_count(policy, n):
  return 0 if policy["kind"] == "improved_threshold" else n


def evaluate_policy(policy, base_score, improved_score, features, args):
  labels = features["labels"]
  pred, improved_called = apply_policy(policy, base_score, improved_score, features)
  n = len(labels)

  tp = int(np.sum((pred == 1) & (labels == 1)))
  tn = int(np.sum((pred == 0) & (labels == 0)))
  fp = int(np.sum((pred == 1) & (labels == 0)))
  fn = int(np.sum((pred == 0) & (labels == 1)))
  acc = float(np.mean(pred == labels))
  macro_f1 = float(f1_score(labels, pred, average="macro", zero_division=0))
  precision = tp / max(tp + fp, 1)
  recall = tp / max(tp + fn, 1)
  improved_calls = int(np.sum(improved_called))
  base_calls = base_call_count(policy, n)
  expected_cost = (
    args.base_call_cost * base_calls
    + args.improved_call_cost * improved_calls
    + args.fp_cost * fp
    + args.fn_cost * fn
  )

  return {
    "policy": policy,
    "accuracy": acc,
    "macro_f1": macro_f1,
    "precision": float(precision),
    "recall": float(recall),
    "tp": tp,
    "tn": tn,
    "fp": fp,
    "fn": fn,
    "base_calls": int(base_calls),
    "improved_calls": improved_calls,
    "improved_call_rate": improved_calls / n,
    "expected_cost": float(expected_cost),
    "expected_cost_per_example": float(expected_cost / n),
    "positive_rate": float(np.mean(pred)),
  }


def make_policy(kind, **params):
  return {"kind": kind, "params": params}


def generate_candidate_policies(args):
  thresholds = frange(args.threshold_start, args.threshold_stop, args.threshold_step)
  coarse_thresholds = frange(0.20, 0.80, 0.05)
  base_thresholds = frange(0.30, 0.70, 0.05)
  keep_thresholds = [0.65, 0.75, 0.85, 0.95, 1.01]
  lower_bounds = [0.20, 0.30, 0.40]
  upper_bounds = [0.60, 0.70, 0.80, 0.90]
  ensemble_weights = [0.25, 0.50, 0.75]
  jaccard_highs = [0.45, 0.55, 0.65, 0.75]
  jaccard_lows = [0.10, 0.20, 0.30]
  rule_jaccards = [0.50, 0.60, 0.70]

  for threshold in thresholds:
    yield make_policy("base_threshold", threshold=threshold)
    yield make_policy("improved_threshold", threshold=threshold)

  for base_weight in ensemble_weights:
    for threshold in thresholds:
      yield make_policy("weighted_ensemble", base_weight=base_weight, threshold=threshold)

  for base_threshold in base_thresholds:
    for rule_jaccard in rule_jaccards:
      yield make_policy("rule_veto", base_threshold=base_threshold, rule_jaccard=rule_jaccard)

  for base_threshold in base_thresholds:
    for improved_threshold in coarse_thresholds:
      for base_keep_threshold in keep_thresholds:
        yield make_policy(
          "improved_veto",
          base_threshold=base_threshold,
          improved_threshold=improved_threshold,
          base_keep_threshold=base_keep_threshold,
        )
        yield make_policy(
          "improved_veto",
          base_threshold=base_threshold,
          improved_threshold=improved_threshold,
          base_keep_threshold=base_keep_threshold,
          rule_jaccard=0.60,
        )

  for base_threshold in [0.40, 0.50, 0.60]:
    for lower in lower_bounds:
      for upper in upper_bounds:
        if lower >= upper:
          continue
        for improved_threshold in coarse_thresholds:
          yield make_policy(
            "uncertain_improved",
            base_threshold=base_threshold,
            lower=lower,
            upper=upper,
            improved_threshold=improved_threshold,
          )
        for base_weight in ensemble_weights:
          for threshold in frange(0.35, 0.65, 0.05):
            yield make_policy(
              "uncertain_weighted",
              base_threshold=base_threshold,
              lower=lower,
              upper=upper,
              base_weight=base_weight,
              threshold=threshold,
            )

  for base_threshold in [0.40, 0.50, 0.60]:
    for jaccard_high in jaccard_highs:
      for base_keep_threshold in keep_thresholds:
        for improved_threshold in coarse_thresholds:
          yield make_policy(
            "segment_veto",
            base_threshold=base_threshold,
            jaccard_high=jaccard_high,
            base_keep_threshold=base_keep_threshold,
            improved_threshold=improved_threshold,
          )
          yield make_policy(
            "segment_veto",
            base_threshold=base_threshold,
            jaccard_high=jaccard_high,
            base_keep_threshold=base_keep_threshold,
            improved_threshold=improved_threshold,
            rule_jaccard=0.60,
          )

  for base_threshold in [0.40, 0.50, 0.60]:
    for jaccard_high in jaccard_highs:
      for jaccard_low in jaccard_lows:
        for improved_threshold in coarse_thresholds:
          yield make_policy(
            "segment_veto_rescue",
            base_threshold=base_threshold,
            jaccard_high=jaccard_high,
            jaccard_low=jaccard_low,
            base_keep_threshold=0.85,
            improved_threshold=improved_threshold,
            rescue_threshold=improved_threshold,
          )


def select_result(results, args):
  if args.selection_metric == "accuracy_band_cost":
    best_accuracy = max(row["accuracy"] for row in results)
    eligible = [
      row for row in results
      if row["accuracy"] >= best_accuracy - args.accuracy_tolerance
    ]
    return sorted(
      eligible,
      key=lambda row: (
        row["improved_call_rate"],
        row["expected_cost_per_example"],
        -row["accuracy"],
        -row["macro_f1"],
      ),
    )[0]

  if args.selection_metric == "accuracy":
    return sorted(
      results,
      key=lambda row: (-row["accuracy"], -row["macro_f1"], row["improved_call_rate"]),
    )[0]

  if args.selection_metric == "macro_f1":
    return sorted(
      results,
      key=lambda row: (-row["macro_f1"], -row["accuracy"], row["improved_call_rate"]),
    )[0]

  if args.selection_metric == "expected_cost":
    return sorted(
      results,
      key=lambda row: (row["expected_cost_per_example"], -row["accuracy"], -row["macro_f1"]),
    )[0]

  raise ValueError(f"Unknown selection metric: {args.selection_metric}")


def policy_label(policy):
  params = ", ".join(f"{key}={value}" for key, value in sorted(policy["params"].items()))
  return f"{policy['kind']}({params})"


def write_predictions(path, ids, pred):
  Path(path).parent.mkdir(parents=True, exist_ok=True)
  with open(path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "Predicted_Is_Paraphrase"])
    for sent_id, value in zip(ids, pred):
      writer.writerow([sent_id, int(value)])


def metric_row_md(row):
  return (
    f"| {policy_label(row['policy'])} "
    f"| {row['accuracy']:.6f} "
    f"| {row['macro_f1']:.6f} "
    f"| {row['precision']:.6f} "
    f"| {row['recall']:.6f} "
    f"| {row['tn']} "
    f"| {row['fp']} "
    f"| {row['fn']} "
    f"| {row['tp']} "
    f"| {row['improved_call_rate']:.4f} "
    f"| {row['expected_cost_per_example']:.6f} |"
  )


def segment_summary_md(name, mask, labels, pred):
  if not np.any(mask):
    return f"| {name} | 0 | n/a | n/a | n/a | n/a |"
  seg_labels = labels[mask]
  seg_pred = pred[mask]
  fp = int(np.sum((seg_pred == 1) & (seg_labels == 0)))
  fn = int(np.sum((seg_pred == 0) & (seg_labels == 1)))
  acc = float(np.mean(seg_pred == seg_labels))
  macro_f1 = float(f1_score(seg_labels, seg_pred, average="macro", zero_division=0))
  return f"| {name} | {int(mask.sum())} | {acc:.6f} | {macro_f1:.6f} | {fp} | {fn} |"


def write_report(path, selected, results, base_score, improved_score, features, args):
  labels = features["labels"]
  selected_pred, _ = apply_policy(selected["policy"], base_score, improved_score, features)
  by_accuracy = sorted(results, key=lambda row: (-row["accuracy"], -row["macro_f1"], row["improved_call_rate"]))
  by_cost = sorted(results, key=lambda row: (row["improved_call_rate"], -row["accuracy"], -row["macro_f1"]))

  best_base = max(
    [row for row in results if row["policy"]["kind"] == "base_threshold"],
    key=lambda row: row["accuracy"],
  )
  best_improved = max(
    [row for row in results if row["policy"]["kind"] == "improved_threshold"],
    key=lambda row: row["accuracy"],
  )

  lines = [
    "# Paraphrase Decision Policy Report",
    "",
    "## Selection",
    "",
    f"- Selection metric: `{args.selection_metric}`",
    f"- Accuracy tolerance: `{args.accuracy_tolerance}`",
    f"- Selected policy: `{policy_label(selected['policy'])}`",
    f"- Policy JSON: `{json.dumps(selected['policy'], sort_keys=True)}`",
    "",
    "## Key Metrics",
    "",
    "| Policy | Acc | Macro F1 | Precision | Recall | TN | FP | FN | TP | Improved Call Rate | Expected Cost / Ex |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    metric_row_md(best_base),
    metric_row_md(best_improved),
    metric_row_md(selected),
    "",
    "## Top Policies By Accuracy",
    "",
    "| Policy | Acc | Macro F1 | Precision | Recall | TN | FP | FN | TP | Improved Call Rate | Expected Cost / Ex |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
  ]
  lines.extend(metric_row_md(row) for row in by_accuracy[:15])
  lines.extend([
    "",
    "## Lowest Improved-Call Policies",
    "",
    "| Policy | Acc | Macro F1 | Precision | Recall | TN | FP | FN | TP | Improved Call Rate | Expected Cost / Ex |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
  ])
  lines.extend(metric_row_md(row) for row in by_cost[:15])

  jaccard = features["jaccard"]
  max_len = features["max_len"]
  suspicious = features["suspicious"]
  segments = [
    ("jaccard < 0.20", jaccard < 0.20),
    ("0.20 <= jaccard < 0.45", (jaccard >= 0.20) & (jaccard < 0.45)),
    ("0.45 <= jaccard < 0.65", (jaccard >= 0.45) & (jaccard < 0.65)),
    ("jaccard >= 0.65", jaccard >= 0.65),
    ("max_len <= 8", max_len <= 8),
    ("8 < max_len <= 16", (max_len > 8) & (max_len <= 16)),
    ("max_len > 16", max_len > 16),
    ("lexical rule suspicious", suspicious),
  ]
  lines.extend([
    "",
    "## Selected Policy Segment Metrics",
    "",
    "| Segment | N | Acc | Macro F1 | FP | FN |",
    "|---|---:|---:|---:|---:|---:|",
  ])
  lines.extend(segment_summary_md(name, mask, labels, selected_pred) for name, mask in segments)
  lines.append("")

  Path(path).parent.mkdir(parents=True, exist_ok=True)
  with open(path, "w") as f:
    f.write("\n".join(lines))


def run_policy_search(args):
  dev_rows = load_paraphrase_data(args.para_dev)
  dev_features = build_features(dev_rows, "dev")
  dev_rows_by_split = {"dev": dev_rows}
  dev_features_by_split = {"dev": dev_features}

  base_scores = score_checkpoint(
    "base",
    args.base_filepath,
    dev_rows_by_split,
    dev_features_by_split,
    args,
    ParaphraseDetectionDataset,
    build_base_eval_args,
  )
  improved_scores = score_checkpoint(
    "improved",
    args.improved_filepath,
    dev_rows_by_split,
    dev_features_by_split,
    args,
    ImprovedParaphraseDetectionDataset,
    build_improved_eval_args,
  )

  base_dev = base_scores["dev"]["probs"]
  improved_dev = improved_scores["dev"]["probs"]

  results = []
  candidates = list(generate_candidate_policies(args))
  for policy in tqdm(candidates, desc="policy-search"):
    results.append(evaluate_policy(policy, base_dev, improved_dev, dev_features, args))

  selected = select_result(results, args)
  print("Selected policy:")
  print(f"  {policy_label(selected['policy'])}")
  print(
    "  dev acc "
    f"{selected['accuracy']:.6f}, macro-F1 {selected['macro_f1']:.6f}, "
    f"FP {selected['fp']}, FN {selected['fn']}, "
    f"improved call rate {selected['improved_call_rate']:.4f}"
  )

  dev_pred, _ = apply_policy(selected["policy"], base_dev, improved_dev, dev_features)
  write_predictions(args.para_dev_out, dev_features["ids"], dev_pred)
  write_report(args.report_out, selected, results, base_dev, improved_dev, dev_features, args)

  test_rows = load_paraphrase_data(args.para_test, split="test")
  test_features = build_features(test_rows, "test")
  test_rows_by_split = {"test": test_rows}
  test_features_by_split = {"test": test_features}
  base_test_scores = score_checkpoint(
    "base",
    args.base_filepath,
    test_rows_by_split,
    test_features_by_split,
    args,
    ParaphraseDetectionDataset,
    build_base_eval_args,
  )
  improved_test_scores = score_checkpoint(
    "improved",
    args.improved_filepath,
    test_rows_by_split,
    test_features_by_split,
    args,
    ImprovedParaphraseDetectionDataset,
    build_improved_eval_args,
  )
  test_pred, _ = apply_policy(
    selected["policy"],
    base_test_scores["test"]["probs"],
    improved_test_scores["test"]["probs"],
    test_features,
  )
  write_predictions(args.para_test_out, test_features["ids"], test_pred)
  return selected


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--base_filepath", type=str, default="10-1e-05-paraphrase.pt")
  parser.add_argument("--improved_filepath", type=str, default=DEFAULT_IMPROVED_CHECKPOINT)
  parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
  parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")
  parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-policy-output.csv")
  parser.add_argument("--para_test_out", type=str, default="predictions/para-test-policy-output.csv")
  parser.add_argument("--report_out", type=str, default="reports/paraphrase_decision_policy_report.md")
  parser.add_argument("--cache_dir", type=str, default="predictions/policy_cache")
  parser.add_argument("--refresh_cache", action="store_true")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--batch_size", type=int, default=128)
  parser.add_argument("--use_gpu", action="store_true")
  parser.add_argument("--multi_gpu", action="store_true")
  parser.add_argument("--gpu_ids", type=str, default=None)

  parser.add_argument("--prompt_template", type=str, default="paraphrase")
  parser.add_argument("--prompt_ensemble_eval", action="store_true")
  parser.add_argument("--prompt_ensemble_templates", type=str, default="paraphrase,duplicate,equivalent")
  parser.add_argument("--bidirectional_eval", action="store_true")
  parser.add_argument("--threshold", type=float, default=None)
  parser.add_argument("--tune_threshold", action="store_true")
  parser.add_argument("--override_checkpoint_eval_args", action="store_true")

  parser.add_argument("--threshold_start", type=float, default=0.05)
  parser.add_argument("--threshold_stop", type=float, default=0.95)
  parser.add_argument("--threshold_step", type=float, default=0.01)
  parser.add_argument(
    "--selection_metric",
    type=str,
    choices=["accuracy_band_cost", "accuracy", "macro_f1", "expected_cost"],
    default="accuracy_band_cost",
  )
  parser.add_argument("--accuracy_tolerance", type=float, default=0.001)
  parser.add_argument("--fp_cost", type=float, default=1.0)
  parser.add_argument("--fn_cost", type=float, default=1.0)
  parser.add_argument("--base_call_cost", type=float, default=0.0)
  parser.add_argument("--improved_call_cost", type=float, default=0.0)
  return parser.parse_args()


if __name__ == "__main__":
  args = get_args()
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  run_policy_search(args)

# !/usr/bin/env python3

"""
Quora paraphrase detection을 위한 평가.

model_eval_paraphrase: 레이블 정보가 있는 dev 및 train dataloader에 적합함.
model_test_paraphrase: 레이블 정보가 없는 test dataloader에 적합.
"""

import torch
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import numpy as np
from sacrebleu.metrics import CHRF
from datasets import (
  SonnetsDataset,
)

TQDM_DISABLE = False


def _batch_paraphrase_logits(batch, model, device, bidirectional=False):
  b_ids = batch['token_ids'].to(device)
  b_mask = batch['attention_mask'].to(device)
  logits = model(b_ids, b_mask)

  if bidirectional:
    rb_ids = batch['reversed_token_ids'].to(device)
    rb_mask = batch['reversed_attention_mask'].to(device)
    reversed_logits = model(rb_ids, rb_mask)
    logits = (logits + reversed_logits) / 2

  return logits.cpu().numpy()


def _preds_from_logits(logits, threshold=None):
  if threshold is None:
    return np.argmax(logits, axis=1).flatten()

  probs = torch.softmax(torch.as_tensor(logits), dim=1).numpy()
  return (probs[:, 1] >= threshold).astype(int).flatten()


@torch.no_grad()
def model_eval_paraphrase(dataloader, model, device, bidirectional=False, threshold=None):
  model.eval()  # Switch to eval model, will turn off randomness like dropout.
  y_true, y_pred, sent_ids = [], [], []
  for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
    b_sent_ids, labels = batch['sent_ids'], batch['labels'].flatten()

    logits = _batch_paraphrase_logits(batch, model, device, bidirectional=bidirectional)
    preds = _preds_from_logits(logits, threshold=threshold)

    y_true.extend(labels.cpu().numpy())
    y_pred.extend(preds)
    sent_ids.extend(b_sent_ids)

  f1 = f1_score(y_true, y_pred, average='macro')
  acc = accuracy_score(y_true, y_pred)

  return acc, f1, y_pred, y_true, sent_ids


@torch.no_grad()
def model_test_paraphrase(dataloader, model, device, bidirectional=False, threshold=None):
  model.eval()  # Switch to eval model, will turn off randomness like dropout.
  y_pred, sent_ids = [], []
  for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
    b_sent_ids = batch['sent_ids']

    logits = _batch_paraphrase_logits(batch, model, device, bidirectional=bidirectional)
    preds = _preds_from_logits(logits, threshold=threshold)

    y_pred.extend(preds)
    sent_ids.extend(b_sent_ids)

  return y_pred, sent_ids


@torch.no_grad()
def tune_paraphrase_threshold(dataloader, model, device, bidirectional=False, start=0.40, stop=0.60, step=0.01):
  model.eval()
  y_true, all_logits = [], []
  for batch in tqdm(dataloader, desc='threshold', disable=TQDM_DISABLE):
    labels = batch['labels'].flatten()
    logits = _batch_paraphrase_logits(batch, model, device, bidirectional=bidirectional)

    y_true.extend(labels.cpu().numpy())
    all_logits.append(logits)

  logits = np.concatenate(all_logits, axis=0)
  y_true = np.asarray(y_true)
  best_threshold, best_acc, best_f1 = 0.5, -1.0, -1.0

  for threshold in np.arange(start, stop + step / 2, step):
    preds = _preds_from_logits(logits, threshold=float(threshold))
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average='macro')
    if acc > best_acc or (acc == best_acc and f1 > best_f1):
      best_threshold, best_acc, best_f1 = float(threshold), acc, f1

  return best_threshold, best_acc, best_f1


def test_sonnet(
    test_path='predictions/generated_sonnets.txt',
    gold_path='data/TRUE_sonnets_held_out.txt'
):
    chrf = CHRF()  # Character n-gram F-score

    # get the sonnets
    generated_sonnets = [x[1] for x in SonnetsDataset(test_path)]
    true_sonnets = [x[1] for x in SonnetsDataset(gold_path)]
    max_len = min(len(true_sonnets), len(generated_sonnets))
    true_sonnets = true_sonnets[:max_len]
    generated_sonnets = generated_sonnets[:max_len]

    # compute chrf
    chrf_score = chrf.corpus_score(generated_sonnets, [true_sonnets])
    return float(chrf_score.score)
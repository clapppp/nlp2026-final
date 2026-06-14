# !/usr/bin/env python3


"""
이 파일은 Quora의 Paraphrase Detection을 위한 Dataset 클래스를 포함한다. 추가 데이터 소스로 훈련시키거나
Quora 데이터셋의 처리 방식(예: 데이터 증강 등)을 변경하려는 경우 이 파일을 수정할 수 있다.
"""

import csv

import re
import torch
from pathlib import Path

from torch.utils.data import Dataset
from transformers import GPT2Tokenizer


PARAPHRASE_PROMPT_TEMPLATES = {
  'paraphrase': 'Is "{sent1}" a paraphrase of "{sent2}"? Answer "yes" or "no": ',
  'duplicate': 'Do these two questions ask the same thing? Question 1: "{sent1}" Question 2: "{sent2}" Answer "yes" or "no": ',
  'equivalent': 'Are the following questions semantically equivalent? Question A: "{sent1}" Question B: "{sent2}" Answer "yes" or "no": ',
}


def preprocess_string(s):
  return ' '.join(s.lower()
                  .replace('.', ' .')
                  .replace('?', ' ?')
                  .replace(',', ' ,')
                  .replace('\'', ' \'')
                  .split())


def _parse_prompt_names(raw_names):
  names = [name.strip() for name in raw_names.split(',') if name.strip()]
  unknown = [name for name in names if name not in PARAPHRASE_PROMPT_TEMPLATES]
  if unknown:
    raise ValueError(f"Unknown paraphrase prompt template(s): {', '.join(unknown)}")
  return names


def get_prompt_template_names(args, ensemble=False):
  if ensemble:
    raw_names = getattr(args, 'prompt_ensemble_templates', 'paraphrase,duplicate,equivalent')
  else:
    raw_names = getattr(args, 'prompt_template', 'paraphrase')
  return _parse_prompt_names(raw_names)


def format_paraphrase_prompt(sent1, sent2, template_name='paraphrase'):
  return PARAPHRASE_PROMPT_TEMPLATES[template_name].format(sent1=sent1, sent2=sent2)


def augment_with_sentence_swaps(dataset):
  augmented = []
  for sent1, sent2, label, sent_id in dataset:
    augmented.append((sent1, sent2, label, sent_id))
    augmented.append((sent2, sent1, label, f'{sent_id}-swap'))
  return augmented


def lexical_jaccard(sent1, sent2):
  words1 = set(sent1.split())
  words2 = set(sent2.split())
  if not words1 and not words2:
    return 0.0
  return len(words1 & words2) / len(words1 | words2)


class ParaphraseDetectionDataset(Dataset):
  def __init__(self, dataset, args):
    self.dataset = dataset
    self.p = args
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.dataset[idx]

  def collate_fn(self, all_data):
    sent1 = [x[0] for x in all_data]
    sent2 = [x[1] for x in all_data]
    labels = torch.LongTensor([x[2] for x in all_data])
    sent_ids = [x[3] for x in all_data]

    prompt_name = get_prompt_template_names(self.p)[0]
    cloze_style_sents = [format_paraphrase_prompt(s1, s2, prompt_name) for (s1, s2) in zip(sent1, sent2)]
    encoding = self.tokenizer(cloze_style_sents, return_tensors='pt', padding=True, truncation=True)

    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])
    example_weights = torch.ones(len(all_data), dtype=torch.float)

    hard_negative_weight = getattr(self.p, 'hard_negative_weight', 1.0)
    hard_negative_jaccard = getattr(self.p, 'hard_negative_jaccard', 0.6)
    if hard_negative_weight > 1.0:
      for i, (s1, s2, label) in enumerate(zip(sent1, sent2, labels)):
        if label.item() == 0 and lexical_jaccard(s1, s2) >= hard_negative_jaccard:
          example_weights[i] = hard_negative_weight

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'labels': labels,
      'example_weights': example_weights,
      'sent_ids': sent_ids
    }

    if getattr(self.p, 'bidirectional_eval', False) or getattr(self.p, 'tune_threshold', False):
      reversed_cloze_style_sents = [format_paraphrase_prompt(s2, s1, prompt_name) for (s1, s2) in zip(sent1, sent2)]
      reversed_encoding = self.tokenizer(reversed_cloze_style_sents, return_tensors='pt', padding=True, truncation=True)
      batched_data['reversed_token_ids'] = torch.LongTensor(reversed_encoding['input_ids'])
      batched_data['reversed_attention_mask'] = torch.LongTensor(reversed_encoding['attention_mask'])

    if getattr(self.p, 'prompt_ensemble_eval', False):
      ensemble_token_ids = []
      ensemble_attention_mask = []
      reversed_ensemble_token_ids = []
      reversed_ensemble_attention_mask = []

      for ensemble_prompt_name in get_prompt_template_names(self.p, ensemble=True):
        ensemble_sents = [
          format_paraphrase_prompt(s1, s2, ensemble_prompt_name) for (s1, s2) in zip(sent1, sent2)
        ]
        ensemble_encoding = self.tokenizer(ensemble_sents, return_tensors='pt', padding=True, truncation=True)
        ensemble_token_ids.append(torch.LongTensor(ensemble_encoding['input_ids']))
        ensemble_attention_mask.append(torch.LongTensor(ensemble_encoding['attention_mask']))

        if getattr(self.p, 'bidirectional_eval', False) or getattr(self.p, 'tune_threshold', False):
          reversed_ensemble_sents = [
            format_paraphrase_prompt(s2, s1, ensemble_prompt_name) for (s1, s2) in zip(sent1, sent2)
          ]
          reversed_ensemble_encoding = self.tokenizer(
            reversed_ensemble_sents, return_tensors='pt', padding=True, truncation=True)
          reversed_ensemble_token_ids.append(torch.LongTensor(reversed_ensemble_encoding['input_ids']))
          reversed_ensemble_attention_mask.append(torch.LongTensor(reversed_ensemble_encoding['attention_mask']))

      batched_data['ensemble_token_ids'] = ensemble_token_ids
      batched_data['ensemble_attention_mask'] = ensemble_attention_mask
      if reversed_ensemble_token_ids:
        batched_data['reversed_ensemble_token_ids'] = reversed_ensemble_token_ids
        batched_data['reversed_ensemble_attention_mask'] = reversed_ensemble_attention_mask

    return batched_data


class ParaphraseDetectionTestDataset(Dataset):
  def __init__(self, dataset, args):
    self.dataset = dataset
    self.p = args
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.dataset[idx]

  def collate_fn(self, all_data):
    sent1 = [x[0] for x in all_data]
    sent2 = [x[1] for x in all_data]
    sent_ids = [x[2] for x in all_data]

    prompt_name = get_prompt_template_names(self.p)[0]
    cloze_style_sents = [format_paraphrase_prompt(s1, s2, prompt_name) for (s1, s2) in zip(sent1, sent2)]

    encoding = self.tokenizer(cloze_style_sents, return_tensors='pt', padding=True, truncation=True)

    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'sent_ids': sent_ids
    }

    if getattr(self.p, 'bidirectional_eval', False) or getattr(self.p, 'tune_threshold', False):
      reversed_cloze_style_sents = [format_paraphrase_prompt(s2, s1, prompt_name) for (s1, s2) in zip(sent1, sent2)]
      reversed_encoding = self.tokenizer(reversed_cloze_style_sents, return_tensors='pt', padding=True, truncation=True)
      batched_data['reversed_token_ids'] = torch.LongTensor(reversed_encoding['input_ids'])
      batched_data['reversed_attention_mask'] = torch.LongTensor(reversed_encoding['attention_mask'])

    if getattr(self.p, 'prompt_ensemble_eval', False):
      ensemble_token_ids = []
      ensemble_attention_mask = []
      reversed_ensemble_token_ids = []
      reversed_ensemble_attention_mask = []

      for ensemble_prompt_name in get_prompt_template_names(self.p, ensemble=True):
        ensemble_sents = [
          format_paraphrase_prompt(s1, s2, ensemble_prompt_name) for (s1, s2) in zip(sent1, sent2)
        ]
        ensemble_encoding = self.tokenizer(ensemble_sents, return_tensors='pt', padding=True, truncation=True)
        ensemble_token_ids.append(torch.LongTensor(ensemble_encoding['input_ids']))
        ensemble_attention_mask.append(torch.LongTensor(ensemble_encoding['attention_mask']))

        if getattr(self.p, 'bidirectional_eval', False) or getattr(self.p, 'tune_threshold', False):
          reversed_ensemble_sents = [
            format_paraphrase_prompt(s2, s1, ensemble_prompt_name) for (s1, s2) in zip(sent1, sent2)
          ]
          reversed_ensemble_encoding = self.tokenizer(
            reversed_ensemble_sents, return_tensors='pt', padding=True, truncation=True)
          reversed_ensemble_token_ids.append(torch.LongTensor(reversed_ensemble_encoding['input_ids']))
          reversed_ensemble_attention_mask.append(torch.LongTensor(reversed_ensemble_encoding['attention_mask']))

      batched_data['ensemble_token_ids'] = ensemble_token_ids
      batched_data['ensemble_attention_mask'] = ensemble_attention_mask
      if reversed_ensemble_token_ids:
        batched_data['reversed_ensemble_token_ids'] = reversed_ensemble_token_ids
        batched_data['reversed_ensemble_attention_mask'] = reversed_ensemble_attention_mask

    return batched_data


def load_paraphrase_data(paraphrase_filename, split='train'):
  paraphrase_data = []
  if split == 'test':
    with open(paraphrase_filename, 'r') as fp:
      for record in csv.DictReader(fp, delimiter='\t'):
        sent_id = record['id'].lower().strip()
        paraphrase_data.append((preprocess_string(record['sentence1']),
                                preprocess_string(record['sentence2']),
                                sent_id))

  else:
    skipped = 0
    with open(paraphrase_filename, 'r') as fp:
      for record in csv.DictReader(fp, delimiter='\t'):
        try:
          sent_id = record['id'].lower().strip()
          paraphrase_data.append((preprocess_string(record['sentence1']),
                                  preprocess_string(record['sentence2']),
                                  int(float(record['is_duplicate'])), sent_id))
        except (KeyError, TypeError, ValueError):
          skipped += 1

  print(f"Loaded {len(paraphrase_data)} {split} examples from {paraphrase_filename}")
  if split != 'test' and skipped:
    print(f"Skipped {skipped} malformed {split} examples from {paraphrase_filename}")
  return paraphrase_data


class SonnetsDataset(Dataset):
  def __init__(self, file_path, append_eos=False):
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.append_eos = append_eos

    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.sonnets = self._load_sonnets(file_path)

  def _load_sonnets(self, file_path):
    """Reads the file and extracts individual sonnets."""
    text = Path(file_path).read_text(encoding='utf-8')

    # Capture the original sonnet number instead of replacing it with 0..n-1.
    matches = list(re.finditer(r'(?m)^\s*(\d+)\s*$', text))
    sonnets = []
    for i, match in enumerate(matches):
      sonnet_id = match.group(1)
      start = match.end()
      end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
      sonnet_text = text[start:end].strip()
      if sonnet_text:
        sonnets.append((sonnet_id, sonnet_text))

    return sonnets

  def __len__(self):
    return len(self.sonnets)

  def __getitem__(self, idx):
    return self.sonnets[idx]

  def collate_fn(self, all_data):
    idx = [example[0] for example in all_data]
    sonnets = [example[1] for example in all_data]
    if self.append_eos:
      sonnets = [
        sonnet if sonnet.endswith(self.tokenizer.eos_token) else f'{sonnet}{self.tokenizer.eos_token}'
        for sonnet in sonnets
      ]

    encoding = self.tokenizer(sonnets, return_tensors='pt', padding=True, truncation=True)
    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'sent_ids': idx
    }

    return batched_data

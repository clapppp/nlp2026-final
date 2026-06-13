'''
소넷 생성을 위한 시작 코드.

실행:
  `python sonnet_generation.py --use_gpu`

trains your SonnetGPT model and writes the required submission files.
SonnetGPT 모델을 훈련하고, 필요한 제출용 파일을 작성한다.
'''

import argparse
import random
import re
import torch
from pathlib import Path

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange

from datasets import (
  SonnetsDataset,
)
from evaluation import test_sonnet, test_sonnet_continuation
from models.gpt2 import GPT2Model

from optimizer import AdamW

TQDM_DISABLE = False


def resolve_device(use_gpu):
  if use_gpu and torch.cuda.is_available():
    return torch.device('cuda')
  if use_gpu and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    return torch.device('mps')
  return torch.device('cpu')


def nonempty_lines(text):
  return [line.strip() for line in text.splitlines() if line.strip()]


def count_nonempty_lines(text):
  return len(nonempty_lines(text))


def _split_long_line(line, max_words_per_line):
  words = line.split()
  if len(words) <= max_words_per_line:
    return [line.strip()]

  chunks = []
  current = []
  phrase_breaks = ('.', '?', '!', ';', ':', ',')
  for word in words:
    current.append(word)
    phrase_boundary = word.rstrip(chr(34) + chr(39) + ')]} ').endswith(phrase_breaks)
    if len(current) >= max_words_per_line or (len(current) >= 6 and phrase_boundary):
      chunks.append(' '.join(current).strip())
      current = []
  if current:
    chunks.append(' '.join(current).strip())
  return chunks


def is_bad_fragment(line):
  stripped = line.strip()
  if not stripped:
    return True
  if not re.search(r'[A-Za-z]', stripped):
    return True
  if re.fullmatch(r'\d+', stripped):
    return True
  return len(sonnet_words(stripped)) <= 2


def merge_bad_fragments(lines):
  merged = []
  pending_prefix = []
  for line in lines:
    if is_bad_fragment(line):
      if merged:
        merged[-1] = f'{merged[-1]} {line}'.strip()
      else:
        pending_prefix.append(line)
      continue
    if pending_prefix:
      line = f'{" ".join(pending_prefix)} {line}'.strip()
      pending_prefix = []
    merged.append(line)

  if pending_prefix and merged:
    merged[-1] = f'{merged[-1]} {" ".join(pending_prefix)}'.strip()
  return merged


def split_line_balanced(line, min_words_per_line):
  words = line.split()
  if len(words) <= 1:
    return None
  if len(words) >= min_words_per_line * 2:
    split_idx = len(words) // 2
    split_idx = max(min_words_per_line, min(split_idx, len(words) - min_words_per_line))
  else:
    split_idx = max(1, len(words) // 2)
  left = ' '.join(words[:split_idx]).strip()
  right = ' '.join(words[split_idx:]).strip()
  if not left or not right:
    return None
  return [left, right]


def merge_to_target_count(lines, target_line_count):
  lines = list(lines)
  while len(lines) > target_line_count:
    merge_idx = min(range(len(lines)), key=lambda idx: len(sonnet_words(lines[idx])))
    if merge_idx == 0:
      neighbor_idx = 1
    elif merge_idx == len(lines) - 1:
      neighbor_idx = merge_idx - 1
    else:
      prev_len = len(sonnet_words(lines[merge_idx - 1]))
      next_len = len(sonnet_words(lines[merge_idx + 1]))
      neighbor_idx = merge_idx - 1 if prev_len <= next_len else merge_idx + 1

    if neighbor_idx < merge_idx:
      lines[neighbor_idx] = f'{lines[neighbor_idx]} {lines[merge_idx]}'.strip()
      del lines[merge_idx]
    else:
      lines[merge_idx] = f'{lines[merge_idx]} {lines[neighbor_idx]}'.strip()
      del lines[neighbor_idx]
  return lines


def expand_to_target_count(lines, target_line_count, min_words_per_line):
  lines = list(lines)
  while len(lines) < target_line_count:
    if not lines:
      break
    split_idx = max(range(len(lines)), key=lambda idx: len(lines[idx].split()))
    split_lines = split_line_balanced(lines[split_idx], min_words_per_line)
    if split_lines is None:
      break
    lines[split_idx:split_idx + 1] = split_lines
  return lines


def format_continuation_lines(
    text, target_line_count, max_words_per_line, min_words_per_line=4, clean_fragments=False):
  if target_line_count <= 0:
    return []
  lines = []
  for line in nonempty_lines(text):
    lines.extend(_split_long_line(line, max_words_per_line))

  if clean_fragments:
    lines = merge_bad_fragments(lines)
    lines = merge_to_target_count(lines, target_line_count)
    lines = expand_to_target_count(lines, target_line_count, min_words_per_line)
  else:
    while len(lines) < target_line_count:
      if not lines:
        break
      split_idx = max(range(len(lines)), key=lambda idx: len(lines[idx].split()))
      words = lines[split_idx].split()
      if len(words) <= 1:
        break
      mid = max(1, len(words) // 2)
      lines[split_idx:split_idx + 1] = [
        ' '.join(words[:mid]).strip(),
        ' '.join(words[mid:]).strip(),
      ]

  return lines[:target_line_count]


def strip_prompt_prefix(decoded_output, prompt_text, prompt_line_count):
  decoded_output = decoded_output.strip()
  prompt_text = prompt_text.strip()
  if decoded_output.startswith(prompt_text):
    return decoded_output[len(prompt_text):].strip()

  prompt_lines = nonempty_lines(prompt_text)[:prompt_line_count]
  decoded_lines = nonempty_lines(decoded_output)
  if not prompt_lines:
    return decoded_output

  for idx, prompt_line in enumerate(prompt_lines):
    if idx >= len(decoded_lines):
      return ''
    if decoded_lines[idx] == prompt_line:
      continue
    if idx == len(prompt_lines) - 1 and decoded_lines[idx].startswith(prompt_line):
      remainder = decoded_lines[idx][len(prompt_line):].strip()
      return '\n'.join([remainder] + decoded_lines[idx + 1:]).strip()
    return '\n'.join(decoded_lines[idx:]).strip()

  return '\n'.join(decoded_lines[len(prompt_lines):]).strip()


def build_sonnet_from_prompt(
    prompt_text, decoded_output, target_lines, prompt_line_count, max_words_per_line,
    min_words_per_line=4, clean_fragments=False):
  if target_lines is None or target_lines <= 0:
    return decoded_output.strip()
  prompt_lines = nonempty_lines(prompt_text)[:prompt_line_count]
  continuation_target = max(target_lines - len(prompt_lines), 0)
  continuation_text = strip_prompt_prefix(decoded_output, prompt_text, prompt_line_count)
  continuation_lines = format_continuation_lines(
    continuation_text, continuation_target, max_words_per_line, min_words_per_line, clean_fragments)
  return '\n'.join(prompt_lines + continuation_lines).strip()


def validate_generated_sonnet(prompt_text, generated_text, target_lines, prompt_line_count):
  prompt_lines = nonempty_lines(prompt_text)[:prompt_line_count]
  generated_lines = nonempty_lines(generated_text)
  issues = []
  if target_lines and len(generated_lines) != target_lines:
    issues.append(f'expected {target_lines} lines, got {len(generated_lines)}')
  if generated_lines[:len(prompt_lines)] != prompt_lines:
    issues.append('prompt lines were not preserved')
  return issues


def reached_valid_stop(decoded_text, target_lines, min_final_line_words=0):
  if not target_lines:
    return False
  lines = nonempty_lines(decoded_text)
  if len(lines) < target_lines:
    return False
  if min_final_line_words <= 0:
    return True
  return len(sonnet_words(lines[target_lines - 1])) >= min_final_line_words


def banned_ngram_tokens(token_ids, prompt_token_count, no_repeat_ngram_size):
  if no_repeat_ngram_size <= 0:
    return set()
  generated_tokens = token_ids[0, prompt_token_count:].tolist()
  if no_repeat_ngram_size == 1:
    return set(generated_tokens)
  if len(generated_tokens) + 1 < no_repeat_ngram_size:
    return set()

  prefix = tuple(generated_tokens[-(no_repeat_ngram_size - 1):])
  banned = set()
  for idx in range(len(generated_tokens) - no_repeat_ngram_size + 1):
    ngram = generated_tokens[idx:idx + no_repeat_ngram_size]
    if tuple(ngram[:-1]) == prefix:
      banned.add(ngram[-1])
  return banned


def normalize_word(word):
  return re.sub(r"[^a-z']", '', word.lower()).strip("'")


def sonnet_words(text):
  return [word for word in (normalize_word(raw_word) for raw_word in text.split()) if word]


def line_last_word(line):
  words = sonnet_words(line)
  return words[-1] if words else ''


def rhyme_key(word, suffix_len=3):
  word = normalize_word(word)
  if not word:
    return ''
  vowels = 'aeiouy'
  for idx in range(len(word) - 1, -1, -1):
    if word[idx] in vowels:
      return word[idx:]
  return word[-suffix_len:]


def ngram_repetition_rate(words, ngram_size=3):
  if ngram_size <= 0 or len(words) < ngram_size:
    return 0.0
  ngrams = [tuple(words[idx:idx + ngram_size]) for idx in range(len(words) - ngram_size + 1)]
  if not ngrams:
    return 0.0
  return 1.0 - (len(set(ngrams)) / len(ngrams))


def build_rerank_reference(sonnet_path, suffix_len=3):
  dataset = SonnetsDataset(sonnet_path)
  vocab = set()
  ending_keys = set()
  line_lengths = []

  for _, sonnet in dataset:
    for line in nonempty_lines(sonnet):
      words = sonnet_words(line)
      vocab.update(words)
      if words:
        ending_keys.add(rhyme_key(words[-1], suffix_len=suffix_len))
        line_lengths.append(len(words))

  return {
    'vocab': vocab,
    'ending_keys': ending_keys,
    'avg_line_length': float(np.mean(line_lengths)) if line_lengths else 9.0,
  }


def rhyme_scheme_score(lines, suffix_len=3):
  rhyme_pairs = [(0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 13)]
  matched = 0
  checked = 0
  for left_idx, right_idx in rhyme_pairs:
    if right_idx >= len(lines):
      continue
    left_key = rhyme_key(line_last_word(lines[left_idx]), suffix_len=suffix_len)
    right_key = rhyme_key(line_last_word(lines[right_idx]), suffix_len=suffix_len)
    if not left_key or not right_key:
      continue
    checked += 1
    if left_key == right_key or left_key.endswith(right_key) or right_key.endswith(left_key):
      matched += 1
  return matched / checked if checked else 0.0


@torch.no_grad()
def continuation_average_logprob(model, tokenizer, prompt_text, full_sonnet, device, prompt_line_count):
  encoding = tokenizer(full_sonnet, return_tensors='pt', padding=False, truncation=True).to(device)
  prompt_encoding = tokenizer(
    '\n'.join(nonempty_lines(prompt_text)[:prompt_line_count]),
    return_tensors='pt',
    padding=False,
    truncation=True,
  ).to(device)

  input_ids = encoding['input_ids']
  attention_mask = encoding['attention_mask']
  if input_ids.shape[1] < 2:
    return -100.0

  logits = model(input_ids, attention_mask)
  log_probs = F.log_softmax(logits[:, :-1], dim=-1)
  labels = input_ids[:, 1:]
  selected_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
  continuation_start = max(prompt_encoding['input_ids'].shape[1] - 1, 0)
  continuation_log_probs = selected_log_probs[:, continuation_start:]
  if continuation_log_probs.numel() == 0:
    return -100.0
  return float(continuation_log_probs.mean().item())


def score_candidate(full_sonnet, prompt_text, args, reference, lm_logprob):
  lines = nonempty_lines(full_sonnet)
  prompt_lines = nonempty_lines(prompt_text)[:args.prompt_line_count]
  continuation_lines = lines[len(prompt_lines):]
  continuation_text = '\n'.join(continuation_lines)
  words = sonnet_words(continuation_text)

  validation_issues = validate_generated_sonnet(prompt_text, full_sonnet, args.target_lines, args.prompt_line_count)
  issue_penalty = float(len(validation_issues))
  line_lengths = [len(sonnet_words(line)) for line in continuation_lines]
  line_length_penalty = 0.0
  if line_lengths:
    for length in line_lengths:
      line_length_penalty += max(0, args.min_line_words - length)
      line_length_penalty += max(0, length - args.max_line_words)
    line_length_penalty /= len(line_lengths)

  repetition_rate = ngram_repetition_rate(words, args.rerank_ngram_size)
  vocab_ratio = 0.0
  if words:
    vocab_ratio = sum(1 for word in words if word in reference['vocab']) / len(words)

  ending_keys = [rhyme_key(line_last_word(line), suffix_len=args.rhyme_suffix_len) for line in continuation_lines]
  ending_ratio = 0.0
  nonempty_endings = [key for key in ending_keys if key]
  if nonempty_endings:
    ending_ratio = sum(1 for key in nonempty_endings if key in reference['ending_keys']) / len(nonempty_endings)

  rhyme_score = rhyme_scheme_score(lines, suffix_len=args.rhyme_suffix_len)
  bad_text_penalty = len(re.findall(r"[\[\]{}<>_=*|`~]|\d", continuation_text)) / max(len(lines), 1)

  return (
    args.rerank_lm_weight * lm_logprob
    - args.rerank_format_weight * issue_penalty
    - args.rerank_line_length_weight * line_length_penalty
    - args.rerank_repetition_weight * repetition_rate
    - args.rerank_bad_text_weight * bad_text_penalty
    + args.rerank_lexicon_weight * vocab_ratio
    + args.rerank_ending_weight * ending_ratio
    + args.rerank_rhyme_weight * rhyme_score
  )

# 재현성을 위한 random seed 고정.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class SonnetGPT(nn.Module):
  """Sonnet 생성을 위해 설계된 여러분의 GPT-2 모델."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

    # 기본적으로, 전체 모델을 fine-tuning한다.
    for param in self.gpt.parameters():
      param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    """
    ParaphraseGPT의 forward pass와 유사하지만, 여기서는 시퀀스의 마지막 토큰뿐만 아니라 시퀀스의 각 토큰에 대한 logit을 생성하려고 한다.
    이를 통해, 마지막 토큰에 대한 다음 토큰의 분포만 학습하는 것이 아니라, 모델은 소네트를 구성하는 자연어 분포를 학습할 수 있다.
    """
    output = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
    sequence_output = output['last_hidden_state']
    return self.gpt.hidden_state_to_token(sequence_output)


  def get_device(self):
    for param in self.gpt.parameters():
      return param.device

  @torch.no_grad()
  def generate(self, encoding, temperature=1.05, top_p=0.9, max_length=180,
               top_k=0, target_lines=14, repetition_penalty=1.05,
               min_line_words=4, max_line_words=12, newline_bias=0.0,
               no_repeat_ngram_size=0, penalize_prompt_tokens=False,
               allow_early_eos=False, min_final_line_words=0):
    """
    top-p sampling 과 softmax temperature를 사용하여 새로운 소넷을 생성한다.

    """
    token_ids = encoding.to(self.get_device())
    attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())
    prompt_token_count = token_ids.shape[1]
    newline_token_ids = self.tokenizer.encode('\n', add_special_tokens=False)

    if token_ids.shape[0] != 1:
      raise ValueError("Sonnet generation expects a single prompt at a time.")
    if temperature <= 0:
      raise ValueError("temperature must be positive.")

    for _ in range(max_length):
      # logits을 구하기 위한 forward pass.
      logits_sequence = self.forward(token_ids, attention_mask)
      logits_last_token = logits_sequence[:, -1, :]

      if repetition_penalty and repetition_penalty != 1.0:
        penalty_start = 0 if penalize_prompt_tokens else prompt_token_count
        penalty_token_ids = set(token_ids[0, penalty_start:].tolist())
        penalty_token_ids.discard(self.tokenizer.eos_token_id)
        for newline_token_id in newline_token_ids:
          penalty_token_ids.discard(newline_token_id)
        for token_id in penalty_token_ids:
          if logits_last_token[0, token_id] < 0:
            logits_last_token[0, token_id] *= repetition_penalty
          else:
            logits_last_token[0, token_id] /= repetition_penalty

      for token_id in banned_ngram_tokens(token_ids, prompt_token_count, no_repeat_ngram_size):
        logits_last_token[0, token_id] = float('-inf')

      decoded_so_far = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist(), skip_special_tokens=True)
      if (
          not allow_early_eos
          and target_lines
          and not reached_valid_stop(decoded_so_far, target_lines, min_final_line_words)
      ):
        logits_last_token[0, self.tokenizer.eos_token_id] = float('-inf')

      current_lines = nonempty_lines(decoded_so_far)
      current_line_words = len(current_lines[-1].split()) if current_lines else 0
      if newline_token_ids:
        if current_line_words >= max_line_words:
          for newline_token_id in newline_token_ids:
            logits_last_token[0, newline_token_id] += newline_bias
        elif current_line_words < min_line_words:
          for newline_token_id in newline_token_ids:
            logits_last_token[0, newline_token_id] -= newline_bias

      logits_last_token = logits_last_token / temperature  # Apply temperature scaling

      if top_k and top_k > 0:
        top_k = min(top_k, logits_last_token.shape[-1])
        kth_values = torch.topk(logits_last_token, top_k).values[:, -1].unsqueeze(-1)
        logits_last_token = logits_last_token.masked_fill(logits_last_token < kth_values, float('-inf'))

      # Convert logits to probabilities
      probs = F.softmax(logits_last_token, dim=-1)

      # Top-p (nucleus) sampling
      if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        top_p_mask = cumulative_probs <= top_p
        top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()  # Shift mask right for proper thresholding
        top_p_mask[..., 0] = True  # Always include the highest probability token
        filtered_probs = sorted_probs * top_p_mask  # Zero out unlikely tokens
        filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)  # Normalize probabilities

        # Sample from filtered distribution
        sampled_index = torch.multinomial(filtered_probs, 1)
        sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)
      else:
        sampled_token = torch.multinomial(probs, 1)

      # Stop if end-of-sequence token is reached
      if sampled_token.item() == self.tokenizer.eos_token_id:
        break

      # Append sampled token
      token_ids = torch.cat([token_ids, sampled_token], dim=1)
      attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1
      )

      decoded_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist(), skip_special_tokens=True)
      if target_lines and reached_valid_stop(decoded_output, target_lines, min_final_line_words):
        return token_ids, decoded_output

    generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist(), skip_special_tokens=True)
    return token_ids, generated_output


def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")


@torch.no_grad()
def generate_sonnets_for_dataset(model, dataset, args, device, print_outputs=False):
  generated_sonnets = []
  validation_issues = []
  use_rerank = args.num_candidates > 1
  reference = build_rerank_reference(args.sonnet_path, suffix_len=args.rhyme_suffix_len) if use_rerank else None
  for sonnet_id, prompt_text in dataset:
    generation_prompt = prompt_text.rstrip() + '\n' if args.prompt_terminal_newline else prompt_text
    encoding = model.tokenizer(generation_prompt, return_tensors='pt', padding=False, truncation=True).to(device)
    candidates = []
    for candidate_idx in range(max(args.num_candidates, 1)):
      temperature = args.temperature
      if args.num_candidates > 1 and args.candidate_temperature_jitter > 0:
        center = (args.num_candidates - 1) / 2
        offset_scale = 0.0 if center == 0 else (candidate_idx - center) / center
        temperature = max(0.1, args.temperature + args.candidate_temperature_jitter * offset_scale)

      _, decoded_output = model.generate(
        encoding['input_ids'],
        temperature=temperature,
        top_p=args.top_p,
        max_length=args.max_generation_length,
        top_k=args.top_k,
        target_lines=args.target_lines,
        repetition_penalty=args.repetition_penalty,
        min_line_words=args.min_line_words,
        max_line_words=args.max_line_words,
        newline_bias=args.newline_bias,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        penalize_prompt_tokens=args.penalize_prompt_tokens,
        allow_early_eos=args.allow_early_eos,
        min_final_line_words=args.min_final_line_words,
      )
      full_sonnet = build_sonnet_from_prompt(
        prompt_text,
        decoded_output,
        args.target_lines,
        args.prompt_line_count,
        args.max_line_words,
        args.min_line_words,
        args.clean_fragments,
      )
      lm_logprob = 0.0
      if use_rerank and args.rerank_lm_weight != 0:
        lm_logprob = continuation_average_logprob(
          model, model.tokenizer, prompt_text, full_sonnet, device, args.prompt_line_count)
      candidate_score = (
        score_candidate(full_sonnet, prompt_text, args, reference, lm_logprob)
        if use_rerank else 0.0
      )
      candidates.append((candidate_score, lm_logprob, full_sonnet))

    _, _, full_sonnet = max(candidates, key=lambda candidate: candidate[0])
    issues = validate_generated_sonnet(prompt_text, full_sonnet, args.target_lines, args.prompt_line_count)
    if issues:
      validation_issues.append((sonnet_id, issues))
    generated_sonnets.append((sonnet_id, f'{full_sonnet}\n\n'))
    if print_outputs:
      print(f'{full_sonnet}\n\n')
  return generated_sonnets, validation_issues


def write_generated_sonnets(generated_sonnets, sonnet_out):
  sonnet_out = Path(sonnet_out)
  sonnet_out.parent.mkdir(parents=True, exist_ok=True)
  with sonnet_out.open("w", encoding="utf-8") as f:
    f.write("--Generated Sonnets-- \n\n")
    for sonnet_id, sonnet_text in generated_sonnets:
      f.write(f"\n{sonnet_id}\n")
      f.write(sonnet_text)


def print_validation_issues(validation_issues):
  if not validation_issues:
    return
  for sonnet_id, issues in validation_issues:
    print(f"validation warning for sonnet {sonnet_id}: {', '.join(issues)}")


def train(args):
  """Sonnet 데이터셋에서 소넷 생성을 위해 GPT-2 훈련.""" 
  device = resolve_device(args.use_gpu)
  # 데이터, 해당 데이터셋 및 데이터로드 생성하기.
  sonnet_dataset = SonnetsDataset(args.sonnet_path, append_eos=True)
  sonnet_dataloader = DataLoader(sonnet_dataset, shuffle=True, batch_size=args.batch_size,
                                 collate_fn=sonnet_dataset.collate_fn)

  # 학습 중 미리보기와 튜닝은 dev prompt만 사용한다.
  dev_sonnet_dataset = SonnetsDataset(args.dev_held_out_sonnet_path)

  args = add_arguments(args)
  model = SonnetGPT(args)
  model = model.to(device)

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr)
  best_selection_score = -1.0

  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0

    for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      # 입력을 가져와서 GPU로 보내기(이 모델을 CPU에서 훈련시키는 것을 권장하지 않는다).
      b_ids, b_mask = batch['token_ids'], batch['attention_mask']
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)

      # 손실, 그래디언트를 계산하고 모델 파라미터 업데이트.
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')  # 시퀀스의 마지막 예측은 무시한다.
      labels = b_ids[:, 1:].contiguous().flatten()  # 레이블을 구성하기 위해 첫번째 토큰을 무시한다.
      loss = F.cross_entropy(logits, labels, ignore_index=model.tokenizer.pad_token_id, reduction='mean')
      loss.backward()
      if args.grad_clip and args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

      if args.max_train_batches and num_batches >= args.max_train_batches:
        break

    train_loss = train_loss / num_batches
    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}.")
    model.eval()
    print('Generating dev output sonnets...')
    generated_sonnets, validation_issues = generate_sonnets_for_dataset(
      model, dev_sonnet_dataset, args, device, print_outputs=False)
    print_validation_issues(validation_issues)
    write_generated_sonnets(generated_sonnets, args.dev_sonnet_out)

    if args.dev_gold_sonnet_path and not args.skip_dev_eval:
      full_chrf = test_sonnet(test_path=args.dev_sonnet_out, gold_path=args.dev_gold_sonnet_path)
      continuation_chrf = test_sonnet_continuation(
        test_path=args.dev_sonnet_out,
        gold_path=args.dev_gold_sonnet_path,
        prompt_line_count=args.prompt_line_count,
      )
      print(f"dev sonnet chrF :: {full_chrf :.3f}")
      print(f"dev continuation chrF :: {continuation_chrf :.3f}")
      selection_score = continuation_chrf
      if selection_score > best_selection_score:
        best_selection_score = selection_score
        save_model(model, optimizer, args, args.best_checkpoint_path)

    save_model(model, optimizer, args, f'{epoch}_{args.filepath}')


@torch.no_grad()
def generate_submission_sonnets(args):
  device = resolve_device(args.use_gpu)
  checkpoint_path = args.checkpoint_path
  if not checkpoint_path:
    checkpoint_path = args.best_checkpoint_path if Path(args.best_checkpoint_path).exists() else f'{args.epochs-1}_{args.filepath}'
  if not Path(checkpoint_path).exists():
    raise FileNotFoundError(
      f'Cannot find sonnet checkpoint {checkpoint_path}. Run training first or pass --checkpoint_path.')
  saved = torch.load(checkpoint_path, map_location=device, weights_only=False)

  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  # held-out 데이터셋 만들기: 처음 3 줄만 있다. 나머지를 채우는 것은 여러분 몫이다!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  generated_sonnets, validation_issues = generate_sonnets_for_dataset(
    model, held_out_sonnet_dataset, args, device, print_outputs=True)
  print_validation_issues(validation_issues)
  write_generated_sonnets(generated_sonnets, args.sonnet_out)

  if args.gold_sonnet_path:
    chrf_score = test_sonnet(test_path=args.sonnet_out, gold_path=args.gold_sonnet_path)
    print(f"sonnet chrF :: {chrf_score :.3f}")
    continuation_chrf = test_sonnet_continuation(
      test_path=args.sonnet_out,
      gold_path=args.gold_sonnet_path,
      prompt_line_count=args.prompt_line_count,
    )
    print(f"sonnet continuation chrF :: {continuation_chrf :.3f}")


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt")
  parser.add_argument("--dev_held_out_sonnet_path", type=str, default="data/sonnets_held_out_dev.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")
  parser.add_argument("--dev_sonnet_out", type=str, default="predictions/generated_sonnets_dev.txt")
  parser.add_argument("--checkpoint_path", type=str, default=None)
  parser.add_argument("--best_checkpoint_path", type=str, default="best-sonnet.pt")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  # Generation parameters.
  parser.add_argument("--temperature", type=float, help="softmax temperature.", default=1.05)
  parser.add_argument("--top_p", type=float, help="Cumulative probability distribution for nucleus sampling.",
                      default=0.9)
  parser.add_argument("--top_k", type=int, help="Limit sampling to the top-k tokens; 0 disables top-k.", default=0)
  parser.add_argument("--max_generation_length", type=int, help="Maximum number of generated tokens.", default=180)
  parser.add_argument("--target_lines", type=int, help="Stop after this many non-empty sonnet lines.", default=14)
  parser.add_argument("--repetition_penalty", type=float, help="Penalty for already generated tokens.", default=1.05)
  parser.add_argument("--prompt_line_count", type=int, help="Number of provided prompt lines to preserve.", default=3)
  parser.add_argument("--min_line_words", type=int, help="Discourage line breaks before this many words.", default=4)
  parser.add_argument("--max_line_words", type=int, help="Encourage line breaks after this many words.", default=12)
  parser.add_argument("--min_final_line_words", type=int,
                      help="Require this many words in the final generated line before early stopping; 0 disables.",
                      default=0)
  parser.add_argument("--newline_bias", type=float, help="Logit bias for newline line-length control.", default=0.0)
  parser.add_argument("--no_repeat_ngram_size", type=int, help="Block repeated continuation n-grams; 0 disables it.",
                      default=0)
  parser.add_argument("--penalize_prompt_tokens", action='store_true')
  parser.add_argument("--allow_early_eos", action='store_true')
  parser.add_argument("--prompt_terminal_newline", action='store_true')
  parser.add_argument("--clean_fragments", dest='clean_fragments', action='store_true', default=True)
  parser.add_argument("--no_clean_fragments", dest='clean_fragments', action='store_false')
  parser.add_argument("--num_candidates", type=int, help="Generate this many candidates per prompt and rerank.",
                      default=1)
  parser.add_argument("--candidate_temperature_jitter", type=float,
                      help="Spread candidate temperatures around --temperature.", default=0.0)
  parser.add_argument("--rerank_lm_weight", type=float, default=0.20)
  parser.add_argument("--rerank_format_weight", type=float, default=10.0)
  parser.add_argument("--rerank_line_length_weight", type=float, default=0.35)
  parser.add_argument("--rerank_repetition_weight", type=float, default=2.0)
  parser.add_argument("--rerank_lexicon_weight", type=float, default=0.6)
  parser.add_argument("--rerank_ending_weight", type=float, default=0.4)
  parser.add_argument("--rerank_rhyme_weight", type=float, default=0.3)
  parser.add_argument("--rerank_bad_text_weight", type=float, default=1.0)
  parser.add_argument("--rerank_ngram_size", type=int, default=3)
  parser.add_argument("--rhyme_suffix_len", type=int, default=3)

  parser.add_argument("--batch_size", help='The training batch size.', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--grad_clip", type=float, help="Gradient clipping max norm; <=0 disables it.", default=1.0)
  parser.add_argument("--max_train_batches", type=int, default=0)
  parser.add_argument("--dev_gold_sonnet_path", type=str, default="data/TRUE_sonnets_held_out_dev.txt")
  parser.add_argument("--gold_sonnet_path", type=str, default=None)
  parser.add_argument("--skip_dev_eval", action='store_true')
  parser.add_argument("--skip_train", action='store_true')
  parser.add_argument("--model_size", type=str, help="The model size as specified on hugging face.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2')

  args = parser.parse_args()
  return args


def add_arguments(args):
  """Add arguments that are deterministic on model size."""
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args


if __name__ == "__main__":
  args = get_args()
  args.filepath = f'{args.epochs}-{args.lr}-sonnet.pt'  # 경로명 저장.
  seed_everything(args.seed)  # 재현성을 위한 random seed 고정.
  if not args.skip_train:
    train(args)
  generate_submission_sonnets(args)

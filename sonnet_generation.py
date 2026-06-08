'''
소넷 생성을 위한 시작 코드.

실행:
  `python sonnet_generation.py --use_gpu`

trains your SonnetGPT model and writes the required submission files.
SonnetGPT 모델을 훈련하고, 필요한 제출용 파일을 작성한다.
'''

import argparse
import random
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
from evaluation import test_sonnet
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


def reshape_to_line_count(text, target_lines, max_words_per_line=10):
  # Format generated text into a sonnet-like fixed number of non-empty lines.
  if target_lines is None or target_lines <= 0:
    return text.strip()

  lines = []
  for line in nonempty_lines(text):
    lines.extend(_split_long_line(line, max_words_per_line))

  while len(lines) < target_lines:
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

  return '\n'.join(lines[:target_lines]).strip()

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
  def generate(self, encoding, temperature=0.7, top_p=0.9, max_length=180,
               top_k=0, target_lines=14, repetition_penalty=1.1):
    """
    top-p sampling 과 softmax temperature를 사용하여 새로운 소넷을 생성한다.

    """
    token_ids = encoding.to(self.get_device())
    attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())

    if token_ids.shape[0] != 1:
      raise ValueError("Sonnet generation expects a single prompt at a time.")
    if temperature <= 0:
      raise ValueError("temperature must be positive.")

    for _ in range(max_length):
      # logits을 구하기 위한 forward pass.
      logits_sequence = self.forward(token_ids, attention_mask)
      logits_last_token = logits_sequence[:, -1, :]

      if repetition_penalty and repetition_penalty != 1.0:
        for token_id in set(token_ids[0].tolist()):
          if logits_last_token[0, token_id] < 0:
            logits_last_token[0, token_id] *= repetition_penalty
          else:
            logits_last_token[0, token_id] /= repetition_penalty

      logits_last_token = logits_last_token / temperature  # Apply temperature scaling

      if top_k and top_k > 0:
        top_k = min(top_k, logits_last_token.shape[-1])
        kth_values = torch.topk(logits_last_token, top_k).values[:, -1].unsqueeze(-1)
        logits_last_token = logits_last_token.masked_fill(logits_last_token < kth_values, float('-inf'))

      # Convert logits to probabilities
      probs = torch.nn.functional.softmax(logits_last_token, dim=-1)

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
      if target_lines and count_nonempty_lines(decoded_output) >= target_lines:
        generated_output = reshape_to_line_count(decoded_output, target_lines)
        return token_ids, generated_output

    generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist(), skip_special_tokens=True)
    generated_output = reshape_to_line_count(generated_output, target_lines)
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


def train(args):
  """Sonnet 데이터셋에서 소넷 생성을 위해 GPT-2 훈련.""" 
  device = resolve_device(args.use_gpu)
  # 데이터, 해당 데이터셋 및 데이터로드 생성하기.
  sonnet_dataset = SonnetsDataset(args.sonnet_path)
  sonnet_dataloader = DataLoader(sonnet_dataset, shuffle=True, batch_size=args.batch_size,
                                 collate_fn=sonnet_dataset.collate_fn)

  # held-out 데이터셋 만들기: 처음 3 줄만 있다. 나머지를 채우는 것은 여러분 몫이다!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  args = add_arguments(args)
  model = SonnetGPT(args)
  model = model.to(device)

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr)

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
    print('Generating several output sonnets...')
    model.eval()
    for batch in held_out_sonnet_dataset:
      encoding = model.tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True).to(device)
      output = model.generate(
        encoding['input_ids'],
        temperature=args.temperature,
        top_p=args.top_p,
        max_length=args.max_generation_length,
        top_k=args.top_k,
        target_lines=args.target_lines,
        repetition_penalty=args.repetition_penalty,
      )
      print(f'{output[1]}\n\n')

    save_model(model, optimizer, args, f'{epoch}_{args.filepath}')


@torch.no_grad()
def generate_submission_sonnets(args):
  device = resolve_device(args.use_gpu)
  checkpoint_path = args.checkpoint_path or f'{args.epochs-1}_{args.filepath}'
  if not Path(checkpoint_path).exists():
    raise FileNotFoundError(
      f'Cannot find sonnet checkpoint {checkpoint_path}. Run training first or pass --checkpoint_path.')
  saved = torch.load(checkpoint_path, weights_only=False)

  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  # held-out 데이터셋 만들기: 처음 3 줄만 있다. 나머지를 채우는 것은 여러분 몫이다!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  generated_sonnets = []
  for batch in held_out_sonnet_dataset:
    sonnet_id = batch[0]
    encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)
    _, decoded_output = model.generate(
      encoding['input_ids'],
      temperature=args.temperature,
      top_p=args.top_p,
      max_length=args.max_generation_length,
      top_k=args.top_k,
      target_lines=args.target_lines,
      repetition_penalty=args.repetition_penalty,
    )
    full_sonnet = f'{decoded_output}\n\n'
    generated_sonnets.append((sonnet_id, full_sonnet))

    print(f'{decoded_output}\n\n')

  sonnet_out = Path(args.sonnet_out)
  sonnet_out.parent.mkdir(parents=True, exist_ok=True)
  with sonnet_out.open("w+", encoding="utf-8") as f:
    f.write(f"--Generated Sonnets-- \n\n")
    for sonnet in generated_sonnets:
      f.write(f"\n{sonnet[0]}\n")
      f.write(sonnet[1])

  if args.gold_sonnet_path:
    chrf_score = test_sonnet(test_path=args.sonnet_out, gold_path=args.gold_sonnet_path)
    print(f"sonnet chrF :: {chrf_score :.3f}")


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")
  parser.add_argument("--checkpoint_path", type=str, default=None)

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  # Generation parameters.
  parser.add_argument("--temperature", type=float, help="softmax temperature.", default=1.2)
  parser.add_argument("--top_p", type=float, help="Cumulative probability distribution for nucleus sampling.",
                      default=0.9)
  parser.add_argument("--top_k", type=int, help="Limit sampling to the top-k tokens; 0 disables top-k.", default=0)
  parser.add_argument("--max_generation_length", type=int, help="Maximum number of generated tokens.", default=180)
  parser.add_argument("--target_lines", type=int, help="Stop after this many non-empty sonnet lines.", default=14)
  parser.add_argument("--repetition_penalty", type=float, help="Penalty for already generated tokens.", default=1.1)

  parser.add_argument("--batch_size", help='The training batch size.', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--grad_clip", type=float, help="Gradient clipping max norm; <=0 disables it.", default=1.0)
  parser.add_argument("--max_train_batches", type=int, default=0)
  parser.add_argument("--gold_sonnet_path", type=str, default=None)
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

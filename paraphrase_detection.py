'''
Paraphrase detection을 위한 시작 코드.

고려 사항:
 - ParaphraseGPT: 여러분이 구현한 GPT-2 분류 모델 .
 - train: Quora paraphrase detection 데이터셋에서 ParaphraseGPT를 훈련시키는 절차.
 - test: Test 절차. 프로젝트 결과 제출에 필요한 파일들을 생성함.

실행:
  `python paraphrase_detection.py --use_gpu`
ParaphraseGPT model을 훈련 및 평가하고, 필요한 제출용 파일을 작성한다.
'''

import argparse
import csv
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import (
  ParaphraseDetectionDataset,
  ParaphraseDetectionTestDataset,
  augment_with_sentence_swaps,
  load_paraphrase_data
)
from evaluation import model_eval_paraphrase, model_test_paraphrase, tune_paraphrase_threshold
from models.gpt2 import GPT2Model

from optimizer import AdamW

TQDM_DISABLE = False


def parse_gpu_ids(gpu_ids):
  if gpu_ids is None:
    return None

  gpu_ids = str(gpu_ids).strip()
  if not gpu_ids or gpu_ids.lower() == 'all':
    return None

  try:
    parsed = [int(gpu_id.strip()) for gpu_id in gpu_ids.split(',') if gpu_id.strip()]
  except ValueError as exc:
    raise ValueError(f"--gpu_ids must be a comma-separated list of integers, got: {gpu_ids}") from exc

  if not parsed:
    return None
  if any(gpu_id < 0 for gpu_id in parsed):
    raise ValueError(f"--gpu_ids must be non-negative CUDA device ids, got: {gpu_ids}")
  return parsed


def resolve_device(args):
  if not args.use_gpu:
    return torch.device('cpu')

  gpu_ids = parse_gpu_ids(getattr(args, 'gpu_ids', None))
  if gpu_ids:
    return torch.device(f'cuda:{gpu_ids[0]}')
  return torch.device('cuda')


def maybe_data_parallel(model, args):
  if not args.use_gpu or not getattr(args, 'multi_gpu', False):
    return model

  available_gpus = torch.cuda.device_count()
  if available_gpus < 2:
    print('Requested multi-GPU training, but fewer than 2 CUDA devices are visible; using single GPU.')
    return model

  device_ids = parse_gpu_ids(getattr(args, 'gpu_ids', None))
  if device_ids is None:
    device_ids = list(range(available_gpus))

  invalid_ids = [gpu_id for gpu_id in device_ids if gpu_id >= available_gpus]
  if invalid_ids:
    raise ValueError(
      f"Requested CUDA device ids {invalid_ids}, but only {available_gpus} CUDA devices are visible.")

  if len(device_ids) < 2:
    print(f'Requested multi-GPU training with device ids {device_ids}; using single GPU.')
    return model

  print(f'Using DataParallel on CUDA devices: {device_ids}')
  return nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])


def unwrap_model(model):
  return model.module if isinstance(model, nn.DataParallel) else model


# Fix the random seed.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class ParaphraseGPT(nn.Module):
  """Paraphrase Detection을 위해 설계된 여러분의 GPT-2 Model."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    # Class order must match Quora labels: 0 -> no, 1 -> yes.
    self.register_buffer('answer_token_ids', torch.tensor([3919, 8505], dtype=torch.long))
    self.classification_head = getattr(args, 'classification_head', False)
    if self.classification_head:
      self.dropout = nn.Dropout(getattr(args, 'hidden_dropout_prob', 0.1))
      self.paraphrase_detection_head = nn.Linear(args.d, 2)

    # 기본적으로, 전체 모델을 finetuning 한다.
    for param in self.gpt.parameters():
      param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    """
    Cloze prompt의 마지막 hidden state에서 "no"와 "yes" 다음 토큰 logit을 예측한다.

    입력은 다음과 같은 구조를 갖는다:

      'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": '

    따라서, 문장의 끝에서 다음 토큰에 대한 예측을 해야 할 것이다.
    훈련이 잘 되었다면, 패러프레이즈인 경우에는 토큰 "yes"(BPE index 8505)가, 
    패러프레이즈가 아닌 경우에는 토큰 "no" (BPE index 3919)가 될 것이다.
    """
    output = self.gpt(input_ids, attention_mask)
    last_token = output['last_token']

    if self.classification_head:
      return self.paraphrase_detection_head(self.dropout(last_token))

    next_token_logits = self.gpt.hidden_state_to_token(last_token)
    return next_token_logits.index_select(dim=1, index=self.answer_token_ids)



def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': unwrap_model(model).state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")


def compute_class_weights(dataset):
  counts = np.bincount([example[2] for example in dataset], minlength=2).astype(np.float32)
  counts = np.maximum(counts, 1.0)
  weights = counts.sum() / (2.0 * counts)
  return torch.tensor(weights, dtype=torch.float)




CHECKPOINT_RUNTIME_ARGS = (
  "para_dev",
  "para_test",
  "para_dev_out",
  "para_test_out",
  "filepath",
  "batch_size",
  "use_gpu",
  "gpu_ids",
)

CHECKPOINT_EVAL_ARGS = (
  "bidirectional_eval",
  "prompt_template",
  "prompt_ensemble_eval",
  "prompt_ensemble_templates",
  "threshold",
  "tune_threshold",
)


def build_checkpoint_eval_args(checkpoint_args, cli_args):
  eval_args = argparse.Namespace(**vars(checkpoint_args))

  for name in CHECKPOINT_RUNTIME_ARGS:
    if hasattr(cli_args, name):
      setattr(eval_args, name, getattr(cli_args, name))

  # Older checkpoints may not contain newer optional eval fields; seed them with parser defaults.
  for name in CHECKPOINT_EVAL_ARGS:
    if not hasattr(eval_args, name) and hasattr(cli_args, name):
      setattr(eval_args, name, getattr(cli_args, name))

  if getattr(cli_args, "override_checkpoint_eval_args", False):
    for name in CHECKPOINT_EVAL_ARGS:
      if hasattr(cli_args, name):
        setattr(eval_args, name, getattr(cli_args, name))

  return eval_args

def train(args):
  """Quora 데이터셋에서 Paraphrase Detection을 위한 GPT-2 훈련."""
  device = resolve_device(args)
  # 데이터, 해당 데이터셋 및 데이터로드 생성하기.
  para_train_data = load_paraphrase_data(args.para_train)
  para_dev_data = load_paraphrase_data(args.para_dev)

  if args.augment_swap:
    para_train_data = augment_with_sentence_swaps(para_train_data)
    print(f"Applied sentence-order swap augmentation: {len(para_train_data)} train examples")

  class_weights = None
  if args.class_weighting:
    class_weights = compute_class_weights(para_train_data).to(device)
    print(f"Using class weights: no={class_weights[0].item():.3f}, yes={class_weights[1].item():.3f}")

  para_train_data = ParaphraseDetectionDataset(para_train_data, args)
  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)

  para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                     collate_fn=para_train_data.collate_fn)
  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                   collate_fn=para_dev_data.collate_fn)

  args = add_arguments(args)
  model = ParaphraseGPT(args)
  model = model.to(device)
  model = maybe_data_parallel(model, args)

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
  best_dev_acc = 0
  epochs_without_improvement = 0

  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0
    for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      # 입력을 가져와서 GPU로 보내기(이 모델을 CPU에서 훈련시키는 것을 권장하지 않는다).
      b_ids, b_mask, labels = batch['token_ids'], batch['attention_mask'], batch['labels'].flatten()
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)
      labels = labels.to(device)

      example_weights = batch.get('example_weights', torch.ones_like(labels, dtype=torch.float)).to(device)

      # 손실, 그래디언트를 계산하고 모델 파라미터 업데이트. 
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      per_example_loss = F.cross_entropy(logits, labels, weight=class_weights, reduction='none')
      loss = (per_example_loss * example_weights).sum() / example_weights.sum()
      loss.backward()
      if args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches

    dev_acc, dev_f1, *_ = model_eval_paraphrase(para_dev_dataloader, model, device)

    if dev_acc > best_dev_acc:
      best_dev_acc = dev_acc
      epochs_without_improvement = 0
      save_model(model, optimizer, args, args.filepath)
    else:
      epochs_without_improvement += 1

    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, dev acc :: {dev_acc :.3f}")
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
  print(f"Loaded model to test from {args.filepath}")
  if getattr(args, "override_checkpoint_eval_args", False):
    print("Using CLI override for paraphrase eval args")

  para_dev_data = load_paraphrase_data(eval_args.para_dev)
  para_test_data = load_paraphrase_data(eval_args.para_test, split="test")

  para_dev_data = ParaphraseDetectionDataset(para_dev_data, eval_args)
  para_test_data = ParaphraseDetectionTestDataset(para_test_data, eval_args)

  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=eval_args.batch_size,
                                   collate_fn=para_dev_data.collate_fn)
  para_test_dataloader = DataLoader(para_test_data, shuffle=False, batch_size=eval_args.batch_size,
                                    collate_fn=para_test_data.collate_fn)

  threshold = getattr(eval_args, "threshold", None)
  bidirectional_eval = getattr(eval_args, "bidirectional_eval", False)
  if getattr(eval_args, "tune_threshold", False):
    threshold, tuned_acc, tuned_f1 = tune_paraphrase_threshold(
      para_dev_dataloader, model, device, bidirectional=bidirectional_eval)
    print(f"best dev threshold :: {threshold :.2f}, acc :: {tuned_acc :.3f}, f1 :: {tuned_f1 :.3f}")

  dev_para_acc, _, dev_para_y_pred, _, dev_para_sent_ids = model_eval_paraphrase(
    para_dev_dataloader, model, device, bidirectional=bidirectional_eval, threshold=threshold)
  print(f"dev paraphrase acc :: {dev_para_acc :.3f}")
  test_para_y_pred, test_para_sent_ids = model_test_paraphrase(
    para_test_dataloader, model, device, bidirectional=bidirectional_eval, threshold=threshold)

  with open(eval_args.para_dev_out, "w+", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "Predicted_Is_Paraphrase"])
    for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
      writer.writerow([p, s])

  with open(eval_args.para_test_out, "w+", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "Predicted_Is_Paraphrase"])
    for p, s in zip(test_para_sent_ids, test_para_y_pred):
      writer.writerow([p, s])

def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
  parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
  parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")
  parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
  parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')
  parser.add_argument("--multi_gpu", action='store_true',
                      help="use torch.nn.DataParallel across multiple visible CUDA devices")
  parser.add_argument("--gpu_ids", type=str, default=None,
                      help="comma-separated CUDA device ids to use, e.g. '0,1,2'; default uses all visible GPUs")
  parser.add_argument("--bidirectional_eval", action='store_true',
                      help="average logits from (sentence1, sentence2) and (sentence2, sentence1) prompts at eval/test time")
  parser.add_argument("--prompt_template", type=str, default="paraphrase",
                      choices=['paraphrase', 'duplicate', 'equivalent'],
                      help="prompt template used for training and single-prompt evaluation")
  parser.add_argument("--prompt_ensemble_eval", action='store_true',
                      help="average logits across multiple paraphrase prompt templates at eval/test time")
  parser.add_argument("--prompt_ensemble_templates", type=str, default="paraphrase,duplicate,equivalent",
                      help="comma-separated prompt template names for --prompt_ensemble_eval")
  parser.add_argument("--threshold", type=float, default=None,
                      help="optional P(yes) threshold for prediction; default uses argmax")
  parser.add_argument("--tune_threshold", action='store_true',
                      help="choose a P(yes) threshold on dev before writing predictions")
  parser.add_argument("--skip_train", action='store_true',
                      help="load the existing checkpoint and only run dev/test prediction")
  parser.add_argument("--override_checkpoint_eval_args", action='store_true',
                      help="use CLI prompt/threshold/bidirectional eval args instead of checkpoint defaults")
  parser.add_argument("--augment_swap", action='store_true',
                      help="train on both (sentence1, sentence2) and (sentence2, sentence1)")
  parser.add_argument("--hard_negative_weight", type=float, default=1.0,
                      help="loss multiplier for high lexical-overlap negative pairs; 1.0 disables it")
  parser.add_argument("--hard_negative_jaccard", type=float, default=0.6,
                      help="Jaccard threshold for hard-negative weighting")
  parser.add_argument("--class_weighting", action='store_true',
                      help="use inverse-frequency class weights in cross entropy")
  parser.add_argument("--classification_head", action='store_true',
                      help="use a learned linear head over GPT-2 last-token hidden state instead of the cloze verbalizer")
  parser.add_argument("--hidden_dropout_prob", type=float, default=0.1,
                      help="dropout used by --classification_head")
  parser.add_argument("--grad_clip", type=float, default=0.0,
                      help="clip gradient norm during training; 0 disables clipping")
  parser.add_argument("--weight_decay", type=float, default=0.0,
                      help="AdamW weight decay")
  parser.add_argument("--early_stopping_patience", type=int, default=0,
                      help="stop training after this many non-improving dev epochs; 0 disables it")

  parser.add_argument("--batch_size", help='training batch size; 128 is safe for GPT-2 on the 96GB Blackwell GPUs', type=int, default=128)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str,
                      help="The model size as specified on hugging face. DO NOT use the xl model.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2')

  args = parser.parse_args()
  return args


def add_arguments(args):
  """모델 크기에 따라 결정되는 인수들을 추가."""
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
  filepath_parts = [str(args.epochs), str(args.lr)]
  if args.classification_head:
    filepath_parts.append('linear')
  if args.augment_swap:
    filepath_parts.append('swap')
  if args.hard_negative_weight > 1.0:
    filepath_parts.append('hardneg')
  if args.class_weighting:
    filepath_parts.append('classweight')
  args.filepath = '-'.join(filepath_parts + ['paraphrase.pt'])  # 경로명 저장.
  seed_everything(args.seed)  # 재현성을 위한 random seed 고정.
  if not args.skip_train:
    train(args)
  test(args)

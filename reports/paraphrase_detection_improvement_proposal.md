# Paraphrase Detection 개선 보고서

작성일: 2026-06-11

## 1. 실험 범위와 제약

본 과제의 Paraphrase Detection은 GPT-2를 이용한 cloze-style 이진 판단 문제로 정의된다. 따라서 본 개선의 목표는 BERT, RoBERTa, DeBERTa 같은 sentence-pair encoder로 모델을 교체하는 것이 아니라, 기존 GPT-2 기반 모델을 유지한 상태에서 fine-tuning 전략을 개선하는 것이다.

초기 구현은 PART-1에서 구현한 `GPT2Model`에 pretrained GPT-2 weight를 로드한 뒤, Quora question pair를 다음과 같은 자연어 prompt로 변환한다.

```text
Is "{sentence1}" a paraphrase of "{sentence2}"? Answer "yes" or "no":
```

모델은 마지막 위치의 hidden state를 GPT-2 vocabulary logit으로 변환하고, `"no"`와 `"yes"` token logit만 비교하여 `0/1`을 예측한다. 즉, 기존 구현은 일반적인 linear binary classifier가 아니라 GPT-2의 next-token prediction 능력을 그대로 활용하는 cloze-style classifier이다.

본 보고서에서 제안하는 개선은 다음 원칙을 따른다.

- GPT-2 backbone을 유지한다.
- 과제에서 제공한 Quora train/dev split만 학습, 튜닝, 평가에 사용한다.
- test-student split의 label은 사용하지 않는다.
- cloze-style yes/no prediction 구조를 기본으로 유지한다.
- 성능 비교는 초기 GPT-2 fine-tuning baseline과 개선된 GPT-2 fine-tuning 구현 사이에서 수행한다.

## 2. 초기 구현 성능

초기 구현은 `gpt2`, 10 epoch, learning rate `1e-5` 설정으로 full fine-tuning하였다.

| 모델 | dev accuracy | macro-F1 | 비고 |
|---|---:|---:|---|
| 초기 GPT-2 cloze baseline | 0.878849 | 0.871748 | 10 epoch, lr=1e-5 |

초기 구현의 dev confusion matrix는 다음과 같다.

| true / pred | 0 | 1 |
|---|---:|---:|
| 0 | 22522 | 3015 |
| 1 | 1883 | 13009 |

이 결과에서 가장 먼저 보이는 문제는 false positive가 false negative보다 많다는 점이다. 즉, 실제로는 paraphrase가 아닌 질문 쌍을 paraphrase라고 과하게 판단하는 경향이 있다. dev 전체에서 positive class가 약 36.8%임에도 모델의 positive prediction 수가 실제 positive support보다 더 많기 때문에, 단순 class weighting으로 positive class를 더 밀어주는 방향은 적절하지 않다.

## 3. 에러 패턴 분석

초기 구현의 dev prediction을 기반으로 `reports/paraphrase_visualization/error_samples.csv`를 확인하였다. 오류는 크게 두 축으로 나뉜다.

### 3.1 High-overlap false positive

False positive 중 상당수는 두 질문의 lexical overlap이 매우 높지만, 실제 의미는 같지 않은 경우였다. 특히 다음과 같은 패턴이 반복적으로 나타났다.

| 오류 유형 | 예시 | 해석 |
|---|---|---|
| 역할 반전 | `female relative to male` vs `male relative to female` | 같은 단어가 등장하지만 비교 방향이 반대다. |
| 행위 주체/대상 반전 | `connect laptop through Android phone` vs `connect Android phone through laptop` | 같은 명사들이 등장하지만 도구와 대상이 뒤집혔다. |
| 출발지/도착지 반전 | `Union Station to Dulles` vs `Dulles to Union Station` | 방향성이 의미를 결정한다. |
| 비교 대상 반전 | `guys girls like` vs `girls guys like` | 성별 단어 overlap만 보고 paraphrase로 오판하기 쉽다. |
| 극성/수량 단어 차이 | `single six was not hit` vs `single six was hit`, `shortest` vs `longest` | 작은 단어 하나가 정답을 바꾼다. |
| 부분 포함 관계 | `CoolPad Note 3 & Note 3 lite` vs `CoolPad Note 3 lite` | 한 질문이 다른 질문의 일부만 묻는다. |

이 패턴은 초기 모델이 의미적 동치성보다 표면적인 단어 겹침에 과하게 의존하고 있음을 보여준다. GPT-2는 decoder-only language model이므로 두 문장 사이의 cross-attention을 직접 수행하는 encoder pair model보다 문장 간 세밀한 alignment를 학습하기 어렵다. 따라서 fine-tuning 단계에서 high-overlap negative를 더 강하게 학습시키는 것이 핵심 개선 방향이 된다.

### 3.2 Low-overlap false negative

False negative는 반대로 lexical overlap은 낮지만 의미적으로는 같은 질문인 경우가 많았다.

| 오류 유형 | 예시 | 해석 |
|---|---|---|
| 의미적 paraphrase | `stay peaceful and happy` vs `easiest way to be happy` | 단어는 다르지만 같은 의도를 묻는다. |
| 동의어/표현 차이 | `laziness` vs `procastination`, `heavy water` vs `deuterium oxide` | lexical match가 낮아도 의미가 같다. |
| 축약 표현 | `music apps without Internet` vs `music without wifi for iPod` | 짧은 query에서는 단어 차이가 크게 보인다. |
| 일반화/구체화 | `learn stock trading` vs `investing in stock market from scratch` | 표현 수준이 달라도 사용자 의도는 유사하다. |
| 철자 오류 | `aciditi` 같은 typo | GPT-2 tokenizer가 의미적 유사성을 안정적으로 잡기 어렵다. |

이 패턴은 hard negative만 강화하면 안 된다는 점을 보여준다. high-overlap negative를 강하게 학습시키는 동시에, low-overlap positive도 충분히 학습시켜야 한다. 그렇지 않으면 모델이 "단어가 많이 겹치면 yes, 적게 겹치면 no"라는 얕은 규칙에서 벗어나지 못한다.

### 3.3 분석에서 도출한 핵심 문제

초기 구현의 핵심 문제는 다음 세 가지로 정리된다.

1. **Lexical overlap shortcut**: 단어가 많이 겹치면 paraphrase로 예측하는 경향이 강하다.
2. **Symmetry sensitivity**: paraphrase relation은 대칭적인데, 학습과 예측은 `(sentence1, sentence2)` 한 방향 prompt에 의존한다.
3. **Verbalizer calibration 문제**: `"yes"`와 `"no"` 두 token logit의 argmax만 사용하므로, positive/negative decision boundary가 dev distribution에 최적으로 맞춰져 있지 않다.

따라서 개선 방향은 모델 교체가 아니라 GPT-2 fine-tuning objective와 데이터 weighting을 바꾸는 것으로 잡았다.

## 4. 개선 방향

분석 결과를 바탕으로 개선 구현은 다음 네 가지를 중심으로 설계한다.

### 4.1 Hard negative 중심 학습

초기 구현의 가장 큰 오류는 high-overlap false positive이다. 따라서 label이 `0`이면서 lexical Jaccard overlap이 높은 pair에 더 큰 loss weight를 부여한다.

기존 구현의 loss는 모든 예시에 동일한 weight를 둔다.

```text
L = CE(logits(sentence1, sentence2), label)
```

개선 구현은 예시별 weight를 적용한다.

```text
L = w_i * CE(logits(sentence1, sentence2), label)
```

여기서 high-overlap negative에 대해서는 `w_i > 1`을 사용한다. 예를 들어 `label = 0`이고 `Jaccard(sentence1, sentence2) >= 0.6`이면 `hard_negative_weight`를 적용한다.

이 개선은 다음 오류를 직접 겨냥한다.

- 단어는 같지만 역할이 반대인 pair
- 방향성이 반대인 pair
- 비교 단어 하나가 다른 pair
- 한 질문이 다른 질문의 부분집합인 pair

### 4.2 Low-overlap positive 보강

High-overlap negative만 강화하면 모델이 전체적으로 conservative해져 positive recall이 낮아질 수 있다. 따라서 label이 `1`이면서 lexical overlap이 낮은 pair도 별도 hard positive로 간주한다.

권장 구현은 다음과 같다.

```text
if label == 1 and Jaccard(sentence1, sentence2) <= tau_pos:
    w_i = low_overlap_positive_weight
```

이 항목은 초기 코드에 있는 `hard_negative_weight`의 반대편 축이다. 목적은 모델이 paraphrase를 단순한 단어 겹침이 아니라 의미적 동치성으로 학습하도록 만드는 것이다.

이 개선은 다음 오류를 직접 겨냥한다.

- 동의어 기반 paraphrase
- 짧은 질문끼리의 의미적 paraphrase
- typo가 포함된 positive pair
- 표현 수준이 다른 질문 pair

### 4.3 Bidirectional consistency

Paraphrase relation은 원칙적으로 대칭적이다.

```text
is_paraphrase(sentence1, sentence2) == is_paraphrase(sentence2, sentence1)
```

초기 구현은 한 방향 prompt만 사용한다. 개선 구현에서는 학습 또는 평가 단계에서 반대 방향 prompt도 사용한다.

평가 단계에서는 두 방향 logit을 평균한다.

```text
logits = (logits(sentence1, sentence2) + logits(sentence2, sentence1)) / 2
```

더 강한 학습 개선으로는 두 방향 모두에 cross entropy를 적용하고, 두 예측 분포가 서로 가까워지도록 consistency regularization을 추가할 수 있다.

```text
L = CE(logits12, y)
  + CE(logits21, y)
  + lambda * KL(softmax(logits12), softmax(logits21))
```

이 방식은 특히 출발지/도착지, 주체/대상, 비교 대상이 바뀌는 high-overlap negative를 더 안정적으로 처리하는 데 도움이 된다.

### 4.4 Prompt robustness와 calibration

초기 구현은 하나의 prompt template에 고정되어 있다. 개선 구현에서는 같은 cloze-style 구조를 유지하되, 여러 prompt wording을 사용한다.

예시:

```text
Is "{sent1}" a paraphrase of "{sent2}"? Answer "yes" or "no":
Do these two questions ask the same thing? Question 1: "{sent1}" Question 2: "{sent2}" Answer "yes" or "no":
Are the following questions semantically equivalent? Question A: "{sent1}" Question B: "{sent2}" Answer "yes" or "no":
```

평가 시에는 여러 prompt의 logit을 평균하여 특정 wording에 대한 편향을 줄인다. 또한 최종 예측은 단순 argmax 대신 dev set에서 `P(yes)` threshold를 조정한다.

초기 구현:

```text
prediction = argmax(no_logit, yes_logit)
```

개선 구현:

```text
prediction = 1 if P(yes) >= threshold else 0
```

초기 모델은 false positive가 더 많으므로, threshold를 0.5보다 약간 높이는 방향이 유리할 가능성이 있다.

## 5. 기존 모델과 개선 구현의 상세 차이

개선 구현은 GPT-2를 버리는 방식이 아니라, 같은 GPT-2 backbone 위에서 fine-tuning 방식과 evaluation policy를 바꾸는 방식이다.

| 항목 | 초기 구현 | 개선 구현 | 기대 효과 |
|---|---|---|---|
| Backbone | `GPT2Model.from_pretrained("gpt2")` | 동일하게 GPT-2 사용 | 과제 조건 유지 |
| 태스크 형식 | cloze-style yes/no | cloze-style yes/no 유지 | 과제의 문제 정의 유지 |
| 출력 방식 | 마지막 hidden state에서 `"no"`, `"yes"` token logit만 비교 | 동일한 verbalizer를 기본으로 사용하되 threshold calibration 적용 | decision boundary 보정 |
| 학습 예시 weight | 모든 예시 동일 weight | high-overlap negative와 low-overlap positive에 가중치 부여 | shortcut 학습 완화 |
| 문장 순서 | `(sentence1, sentence2)` 한 방향 | `(sentence1, sentence2)`와 `(sentence2, sentence1)` 모두 활용 | 대칭성 반영 |
| Prompt | 단일 prompt | 여러 cloze prompt template 평가 또는 학습 | prompt wording 편향 감소 |
| Loss | plain cross entropy | weighted cross entropy, optional consistency regularization | 오류 유형별 학습 신호 강화 |
| Regularization | 기본 dropout 외 별도 안정화 약함 | weight decay, gradient clipping, early stopping, optional R-Drop/SMART/FreeLB | overfitting과 불안정한 update 완화 |
| Evaluation | dev accuracy 중심 | accuracy, macro-F1, confusion matrix, Jaccard bucket, length bucket, error samples | 개선 원인 분석 가능 |
| 최종 예측 | argmax | threshold-tuned prediction, optional prompt/bidirectional ensemble | false positive/false negative trade-off 조정 |

### 5.1 Loss function 차이

초기 구현은 cross entropy만 사용한다.

```text
L_base = CE(logits12, y)
```

개선 구현의 기본형은 weighted cross entropy이다.

```text
L_weighted = w_i * CE(logits12, y)
```

여기에 bidirectional 학습을 추가하면 다음과 같다.

```text
L_bidir = w_i * CE(logits12, y)
        + w_i * CE(logits21, y)
```

그리고 consistency regularization까지 추가하면 다음 형태가 된다.

```text
L_total = L_bidir
        + lambda * KL(softmax(logits12), softmax(logits21))
```

이 차이는 단순한 hyperparameter tuning이 아니라 모델이 학습하는 inductive bias 자체를 바꾼다. 초기 구현은 "주어진 한 방향 prompt에서 yes/no를 맞히는 것"만 학습하지만, 개선 구현은 "문장 순서가 바뀌어도 같은 관계 판단을 내리는 것"까지 학습한다.

### 5.2 데이터 사용 방식 차이

초기 구현은 제공된 train pair를 그대로 사용한다. 개선 구현은 label과 lexical overlap을 기준으로 예시의 중요도를 다르게 준다.

| 예시 유형 | 초기 구현 | 개선 구현 |
|---|---|---|
| 일반 negative | weight 1.0 | weight 1.0 |
| 일반 positive | weight 1.0 | weight 1.0 |
| high-overlap negative | weight 1.0 | `hard_negative_weight` 적용 |
| low-overlap positive | weight 1.0 | `low_overlap_positive_weight` 적용 |
| sentence-swapped pair | 사용하지 않음 | augmentation 또는 bidirectional loss로 활용 |

이 방식은 외부 데이터를 쓰지 않으면서도 train split 안에서 모델이 어려워하는 유형을 더 자주, 더 강하게 학습하도록 만든다.

### 5.3 평가 정책 차이

초기 구현은 raw logit argmax를 그대로 사용한다. 개선 구현은 dev set을 이용해 threshold를 선택하고, 그 threshold를 dev/test prediction 생성에 동일하게 적용한다.

이 차이는 중요하다. 모델의 representation이 좋아져도 decision threshold가 맞지 않으면 false positive가 계속 많을 수 있다. 특히 현재 초기 구현은 positive를 과하게 예측하는 경향이 있으므로, threshold tuning은 학습을 다시 하지 않고도 성능을 개선할 수 있는 가장 비용이 낮은 방법이다.

### 5.4 보고서에서 강조할 차이점

최종 보고서에서는 개선 구현을 "더 큰 모델을 썼다"가 아니라 다음과 같이 설명하는 것이 좋다.

> 초기 구현은 GPT-2의 next-token distribution에서 `"yes"`와 `"no"` logit을 비교하는 cloze-style baseline이다. 개선 구현은 같은 GPT-2 cloze framework를 유지하되, error analysis에서 관찰된 lexical-overlap shortcut을 줄이기 위해 high-overlap negative와 low-overlap positive에 서로 다른 학습 가중치를 부여하고, paraphrase relation의 대칭성을 반영하기 위해 bidirectional prompt를 학습 및 평가에 활용하였다. 또한 prompt wording과 yes/no decision boundary에 대한 민감도를 줄이기 위해 prompt ensemble과 threshold calibration을 적용하였다.

이 설명은 과제 조건을 유지하면서도 초기 구현과 개선 구현의 연구적 차이를 분명히 보여준다.

## 6. 현재 개선 결과와 비교 평가 계획

`reports/paraphrase_visualization/summary.md`에 기록된 현재 분석 결과 기준으로, 개선된 prediction은 다음 성능을 보인다.

| 모델 | dev accuracy | macro-F1 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|
| 초기 GPT-2 cloze baseline | 0.878849 | 0.871748 | 22522 | 3015 | 1883 | 13009 |
| 개선 GPT-2 fine-tuning 구현 | 0.903065 | 0.896486 | 23351 | 2186 | 1733 | 13159 |

성능 변화는 다음과 같다.

| 지표 | 변화 |
|---|---:|
| accuracy | +0.024216 |
| macro-F1 | +0.024738 |
| false positive | -829 |
| false negative | -150 |
| true negative | +829 |
| true positive | +150 |

가장 큰 개선은 false positive 감소에서 나온다. 이는 error analysis에서 발견한 high-overlap false positive 문제와 직접 연결된다. 즉, 개선 방향이 단순히 전체 accuracy를 우연히 올린 것이 아니라, 초기 구현의 주된 오류 유형을 겨냥해 실제로 그 오류를 줄였다고 해석할 수 있다.

최종 비교 평가는 다음 항목을 포함해야 한다.

1. 초기 checkpoint와 개선 checkpoint의 dev accuracy/macro-F1 비교
2. confusion matrix 비교
3. false positive와 false negative 변화량 비교
4. Jaccard overlap bucket별 accuracy 비교
5. sentence length bucket별 accuracy 비교
6. 대표 error sample의 정성 분석
7. dev에서 선택한 threshold와 test prediction 생성 시 동일 threshold 사용 여부 기록

## 7. 권장 실험 순서

### 7.1 재학습 없는 실험

먼저 기존 checkpoint를 그대로 두고 evaluation policy만 바꾼다.

```bash
python paraphrase_detection.py \
  --use_gpu \
  --skip_train \
  --bidirectional_eval \
  --prompt_ensemble_eval \
  --tune_threshold \
  --override_checkpoint_eval_args
```

이 실험은 학습 비용이 없고, 현재 관찰된 prompt sensitivity와 decision boundary 문제를 빠르게 확인할 수 있다.

### 7.2 GPT-2 small 개선 학습

다음으로 GPT-2 small에서 hard example weighting과 안정화 설정을 적용한다.

```bash
python paraphrase_detection.py \
  --use_gpu \
  --multi_gpu \
  --gpu_ids 1,2,3 \
  --hard_negative_weight 2.0 \
  --hard_negative_jaccard 0.6 \
  --augment_swap \
  --epochs 10 \
  --batch_size 512 \
  --lr 1e-5 \
  --grad_clip 1.0 \
  --weight_decay 0.01 \
  --early_stopping_patience 2
```

추가 구현이 가능하다면 `low_overlap_positive_weight`를 도입해 low-overlap positive에 별도 weight를 준다.

### 7.3 GPT-2 medium 확장

GPT-2 small에서 가장 좋은 설정을 찾은 뒤에만 `gpt2-medium`으로 확장한다.

```bash
python paraphrase_detection.py \
  --use_gpu \
  --multi_gpu \
  --gpu_ids 1,2,3 \
  --model_size gpt2-medium \
  --hard_negative_weight 2.0 \
  --hard_negative_jaccard 0.6 \
  --augment_swap \
  --epochs 8 \
  --batch_size 128 \
  --lr 5e-6 \
  --grad_clip 1.0 \
  --weight_decay 0.01 \
  --early_stopping_patience 2
```

모델 크기 확장은 마지막 단계로 두는 것이 좋다. 먼저 GPT-2 small에서 error pattern을 줄이는 전략이 실제로 맞는지 확인해야 한다.

## 8. 레퍼런스

본 개선 방향은 다음 연구와 연결된다.

| 주제 | 레퍼런스 | 본 프로젝트에서의 사용 |
|---|---|---|
| GPT 계열 pretraining 후 fine-tuning | Radford et al., *Improving Language Understanding by Generative Pre-Training* | GPT-2 fine-tuning 기반 접근의 출발점 |
| GPT-2 | Radford et al., *Language Models are Unsupervised Multitask Learners* | GPT-2 decoder-only LM을 downstream task에 활용하는 근거 |
| Cloze-style prompt classification | Schick and Schuetze, *Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference* | yes/no verbalizer 기반 분류 설계 근거 |
| Prompt-based fine-tuning | Gao et al., *Making Pre-trained Language Models Better Few-shot Learners* | prompt wording과 verbalizer 선택의 중요성 |
| Calibration | Guo et al., *On Calibration of Modern Neural Networks* | threshold/temperature calibration 필요성 |
| R-Drop | Liang et al., *R-Drop: Regularized Dropout for Neural Networks* | 두 dropout pass의 분포 일관성 regularization |
| SMART | Jiang et al., *SMART: Robust and Efficient Fine-Tuning for Pre-trained Natural Language Models* | pretrained LM fine-tuning 안정화 |
| FreeLB | Zhu et al., *FreeLB: Enhanced Adversarial Training for Natural Language Understanding* | embedding perturbation 기반 robust fine-tuning |
| LoRA | Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* | GPT-2를 유지한 parameter-efficient fine-tuning 후보 |
| ReFT | Wu et al., *ReFT: Representation Finetuning for Language Models* | GPT-2 representation intervention 기반 개선 후보 |

참고 링크:

- GPT-1: https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
- GPT-2: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- PET: https://arxiv.org/abs/2001.07676
- LM-BFF: https://arxiv.org/abs/2012.15723
- Calibration: https://arxiv.org/abs/1706.04599
- R-Drop: https://arxiv.org/abs/2106.14448
- SMART: https://arxiv.org/abs/1911.03437
- FreeLB: https://arxiv.org/abs/1909.11764
- LoRA: https://arxiv.org/abs/2106.09685
- ReFT: https://arxiv.org/abs/2404.03592

## 9. 결론

초기 GPT-2 cloze baseline은 dev accuracy 0.878849로 이미 fine-tuning 효과를 보였지만, error sample 분석 결과 lexical overlap shortcut이 뚜렷했다. 특히 단어가 거의 같지만 의미 역할이 바뀐 high-overlap negative를 paraphrase로 잘못 판단하는 false positive가 많았다. 반대로 단어는 적게 겹치지만 의미가 같은 low-overlap positive를 놓치는 false negative도 존재했다.

따라서 개선 방향은 GPT-2를 다른 encoder 모델로 교체하는 것이 아니라, GPT-2 fine-tuning 과정에서 어려운 예시에 더 강한 학습 신호를 주고, paraphrase relation의 대칭성을 반영하며, prompt와 threshold에 대한 민감도를 줄이는 것으로 정했다.

이 방향은 과제 조건을 유지하면서도 초기 구현과 명확히 구분된다. 초기 구현이 단일 prompt, 단일 방향, 동일 weight, argmax prediction에 의존했다면, 개선 구현은 error analysis에 기반한 weighted fine-tuning, bidirectional evaluation/training, prompt ensemble, threshold calibration을 사용한다. 현재 분석 결과 기준으로 개선 구현은 dev accuracy 0.903065, macro-F1 0.896486을 기록하여 초기 baseline 대비 accuracy를 약 2.42%p 높였고, false positive를 829개 줄였다.

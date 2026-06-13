# GPT-2 아키텍처의 밑바닥 구현과 감정 분석 Fine-tuning: SST·CFIMDB 데이터셋에서 last-linear-layer와 full-model 미세조정 비교

**프로젝트 유형:** 지정 주제 (GPT 모델 구축과 실험) — PART-I

**팀원**

| 이름 | 이메일 | 역할 |
|---|---|---|
| 박수현 | ssantta999@gmail.com | PART-I (GPT-2 구현, AdamW, 감정 분석) |
| (팀원 2) | (이메일) | PART-II |
| (팀원 3) | (이메일) | PART-II |

**GitHub Repository:** https://github.com/clapppp/nlp2026-final

---

## 초록 (Abstract)

본 연구는 OpenAI의 GPT-2 언어 모델을 PyTorch 기반으로 밑바닥부터 구현하고, 이를 두 가지 감정 분석 데이터셋에 미세조정(fine-tuning)하여 분류 모델로 전환하는 과정을 다룬다. 구체적으로 (1) Causal Self-Attention, Pre-LayerNorm 기반 Transformer 디코더 블록, 전체 GPT-2 모델의 누락된 코드 블록을 직접 구현하고, (2) 가중치 감쇠(weight decay)를 분리한 AdamW 옵티마이저의 `step()` 함수를 구현하였으며, (3) 마지막 토큰 표현(last-token representation)에 분류 헤드를 부착한 `GPT2SentimentClassifier`로 SST(5-class)와 CFIMDB(binary) 데이터셋을 미세조정하였다. 구현의 정합성은 HuggingFace 공식 GPT-2 가중치와의 출력 비교(`sanity_check.py`) 및 참조 가중치와의 수치 비교(`optimizer_test.py`)로 검증하였다. 또한 GPT-2를 동결한 채 선형 분류 헤드만 학습하는 `last-linear-layer` 방식과 전체 파라미터를 갱신하는 `full-model` 방식을 비교하여, 사전학습된 표현(pretrained representation)의 전이 가능성과 전체 미세조정의 효과를 정량적으로 분석하였다.

---

## 1. 서론 (Introduction)

### 1.1 연구의 배경

GPT(Generative Pretrained Transformer) 계열 모델은 "대규모 비지도 사전학습(pre-training) 후 후속 태스크에 대한 미세조정(fine-tuning)"이라는 패러다임을 정립하였다. GPT-1(Radford et al., 2018)은 Transformer 기반 디코더를 단방향(unidirectional) 언어 모델로 사전학습한 뒤, 적은 양의 라벨 데이터만으로도 다양한 NLP 태스크에서 큰 성능 향상을 얻을 수 있음을 보였다. 후속 모델인 GPT-2(Radford et al., 2019)는 모델과 데이터의 규모를 확장하여, 명시적인 태스크별 미세조정 없이도 번역·질의응답·요약 등에서 뛰어난 zero-shot 성능을 보여주었다.

이러한 모델이 강력한 이유는, 가공되지 않은 원시 텍스트로부터 일반적인 언어적 특성과 문맥 패턴을 먼저 학습하기 때문이다. 그 결과 모델은 감정 분석과 같은 후속 응용 분야에 두루 활용될 수 있는 포괄적인 "언어 이해" 능력을 갖추게 된다.

### 1.2 기존 접근의 한계와 본 연구의 동기

GPT-2를 단순히 라이브러리(예: HuggingFace `transformers`)를 통해 불러와 사용하는 것만으로는, 내부에서 Self-Attention이 어떻게 미래 토큰을 차단하는지, Pre-LayerNorm 구조가 어떻게 잔차 연결(residual connection)과 결합되는지, AdamW가 가중치 감쇠를 어떻게 분리하여 적용하는지 등 핵심 동작 원리를 체득하기 어렵다. 본 연구는 이러한 핵심 구성 요소를 직접 구현함으로써 모델의 동작을 정확히 이해하고, 구현의 정합성을 공식 모델과의 수치 비교로 검증하는 것을 목표로 한다.

### 1.3 본 연구의 기여

- **GPT-2 핵심 모듈의 직접 구현:** Causal Self-Attention의 scaled dot-product 및 causal masking, Pre-LayerNorm Transformer 블록, 전체 GPT-2 forward 경로를 구현하고 HuggingFace 공식 가중치 출력과 허용 오차(atol=0.1) 내에서 일치함을 확인하였다.
- **AdamW 옵티마이저 구현:** 1·2차 모멘트 추정, 편향 보정(bias correction), 가중치 감쇠 분리(decoupled weight decay)를 구현하고 참조 가중치와 atol=1e-6 수준으로 일치함을 확인하였다.
- **두 가지 미세조정 전략의 정량 비교:** SST와 CFIMDB 두 데이터셋에 대해 `last-linear-layer`와 `full-model` 방식을 비교하고, 기준 모델(baseline)과의 정확도 차이를 분석하였다.

### 1.4 보고서의 구성

2장에서 관련 연구를 정리하고, 3장에서 GPT-2 아키텍처와 AdamW, 감정 분류 헤드의 제안 방법론 및 구현을 수식과 함께 기술한다. 4장에서 데이터셋·평가 방법·실험 설정 및 결과를 제시하고, 5장에서 결론과 향후 연구 방향을 논의한다.

---

## 2. 관련 연구 (Related Works)

**Transformer 디코더와 GPT 계열.** GPT-1(Radford et al., 2018)은 Transformer 디코더를 좌→우 단방향 언어 모델로 사전학습하여, 후속 태스크에서 미세조정만으로 강력한 성능을 달성했다. GPT-2(Radford et al., 2019)는 동일한 디코더-only 구조를 유지하면서 파라미터(117M~1542M)와 학습 데이터 규모를 확장하여, 태스크별 미세조정 없이도 zero-shot 일반화가 가능함을 보였다. 본 연구가 구현하는 모델은 117M 규모의 GPT-2(`gpt2`)로, 동일한 디코더-only·Pre-LayerNorm 구조를 따른다.

**감정 분석과 SST.** Stanford Sentiment Treebank(Socher et al., 2013)는 영화 리뷰에서 추출한 11,855개의 단일 문장을 Stanford parser로 파싱하여 215,154개의 고유 구문(phrase)을 만들고, 각 구문에 5단계(very negative ~ very positive)의 감정 레이블을 부여한 데이터셋이다. 본 연구는 이 데이터셋의 문장 단위 5-class 분류와, 이진(binary) 감정 레이블을 가진 CFIMDB(Compact Fine-grained IMDB)를 사용한다.

**AdamW.** 본 연구가 구현하는 AdamW는 Adam의 적응적 학습률에 가중치 감쇠를 그래디언트 갱신과 분리하여(decoupled) 적용하는 방식으로, 정규화의 효과를 학습률 스케줄과 독립적으로 동작하게 한다. 기존 연구와의 차별점은, 본 연구가 라이브러리 옵티마이저를 사용하는 대신 `step()` 함수를 직접 구현하여 참조 구현과 수치적으로 동일함을 검증한다는 점이다.

---

## 3. 제안 방법론 (Proposed Approach)

> **직접 작성한 코드 명시.** 제공된 스타터 코드의 빈 블록 중, 본 팀이 직접 구현한 부분은 다음과 같다: `modules/attention.py`의 `CausalSelfAttention.attention()`, `modules/gpt2_layer.py`의 `GPT2Layer.forward()` 및 `add()`, `models/gpt2.py`의 forward 경로, `optimizer.py`의 `AdamW.step()`, `classifier.py`의 `GPT2SentimentClassifier`(분류 헤드 및 forward)와 학습 루프의 누락 블록. 클래스 골격·설정(`config.py`)·유틸리티(`utils.py`)·테스트 스크립트(`sanity_check.py`, `optimizer_test.py`)는 과제에서 제공된 것을 그대로 사용하였다.

### 3.1 Causal Self-Attention

입력 은닉 상태 $X \in \mathbb{R}^{B \times T \times H}$에 대해 query/key/value를 선형 변환으로 사영하고, $h$개의 헤드로 분할한다($d_k = H/h$). 각 헤드에서 scaled dot-product attention을 계산한다:

$$
\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M_{\text{causal}} + M_{\text{pad}}\right) V
$$

여기서 인과 마스크 $M_{\text{causal}}$는 상삼각(strictly upper-triangular) 위치를 $-\infty$로 설정하여 각 위치가 미래 토큰을 참조하지 못하도록 한다:

$$
(M_{\text{causal}})_{ij} = \begin{cases} -\infty & j > i \\ 0 & j \le i \end{cases}
$$

$M_{\text{pad}}$는 패딩 위치를 큰 음수로 만들어 softmax 이후 가중치가 0에 수렴하게 한다. 구현에서는 `torch.triu(..., diagonal=1)`로 인과 마스크를 생성한 뒤 `masked_fill`로 적용하고, 패딩 마스크는 점수에 가산한다. 정규화된 attention 가중치에 dropout을 적용한 후 $V$와 곱하여 컨텍스트 벡터를 얻고, 멀티헤드를 다시 $H$ 차원으로 결합한다.

### 3.2 Pre-LayerNorm Transformer 블록

GPT-2는 각 서브레이어 **이전**에 Layer Normalization을 적용하는 Pre-LayerNorm 구조를 사용한다. 블록의 연산은 다음과 같다:

$$
\begin{aligned}
H' &= H + \text{Dropout}\big(W_o \cdot \text{Attention}(\text{LN}_1(H))\big) \\
H'' &= H' + \text{Dropout}\big(W_2 \cdot \text{GELU}(W_1 \cdot \text{LN}_2(H'))\big)
\end{aligned}
$$

즉 (LN → Attention → Dense → Dropout → 잔차합) 후 (LN → FFN → Dropout → 잔차합) 순서로 진행된다. 헬퍼 `add(input, output, dense, dropout)`는 `input + dropout(dense(output))`을 수행하며, 이 단계에서는 Layer Normalization을 적용하지 않는다(LN은 서브레이어 입력에서 이미 적용됨).

### 3.3 GPT-2 모델

토큰 임베딩과 학습 가능한 위치 임베딩(learnable positional embedding)을 더한 뒤 $L$개의 `GPT2Layer`를 통과시키고, 마지막에 최종 Layer Normalization을 적용한다. 감정 분류에는 시퀀스의 **마지막 토큰**의 은닉 표현을 문장 표현으로 사용한다. 구현 정합성은 HuggingFace `GPT2Model`과 동일 입력에 대한 마지막 은닉 상태를 비교하여(`atol=0.1, rtol=0.01`) 검증하였다.

### 3.4 AdamW 옵티마이저

각 파라미터 $\theta$에 대해 1차/2차 모멘트를 지수이동평균으로 갱신한다:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

편향 보정을 반영한 스텝 크기는 다음과 같다:

$$
\text{step\_size} = \alpha \cdot \frac{\sqrt{1-\beta_2^{\,t}}}{1-\beta_1^{\,t}}, \qquad
\theta_t \leftarrow \theta_{t-1} - \text{step\_size}\cdot\frac{m_t}{\sqrt{v_t}+\epsilon}
$$

마지막으로, 가중치 감쇠를 그래디언트 갱신과 **분리하여** 적용한다(AdamW의 핵심):

$$
\theta_t \leftarrow \theta_t - \alpha\,\lambda\,\theta_t
$$

여기서 $\lambda$는 weight decay 계수다. 구현은 `optimizer_test.py`의 1,000 스텝 학습 결과를 참조 가중치(`optimizer_test.npy`)와 비교하여 `atol=1e-6, rtol=1e-4` 수준으로 일치함을 확인하였다.

### 3.5 감정 분류 헤드와 미세조정 전략

`GPT2SentimentClassifier`는 GPT-2가 출력한 마지막 토큰 표현 $z \in \mathbb{R}^{H}$에 dropout과 선형 사영을 적용하여 클래스 로짓을 산출한다:

$$
\text{logits} = W_{\text{cls}} \cdot \text{Dropout}(z) + b_{\text{cls}}, \qquad W_{\text{cls}} \in \mathbb{R}^{C \times H}
$$

SST는 $C=5$, CFIMDB는 $C=2$이다. 학습은 두 가지 전략으로 수행한다:

- **last-linear-layer:** GPT-2 파라미터를 동결(`requires_grad=False`)하고 분류 헤드만 학습한다. 사전학습된 표현의 전이 가능성을 평가한다.
- **full-model:** GPT-2 전체 파라미터를 함께 갱신한다. 태스크에 맞춰 표현 자체를 적응시킨다.

### 3.6 기준 모델 (Baseline)

지정 주제 과제에서 제공한 기준 모델은 본 구현이 도달해야 할 표준 성능을 정의한다. 과제 명세에 따른 dev 정확도 기준값은 다음과 같다:

| 데이터셋 | last-linear-layer | full-model |
|---|---|---|
| SST (5-class) | 0.462 | 0.513 |
| CFIMDB (binary) | 0.861 | 0.976 |

---

## 4. 실험 (Experiments)

### 4.1 데이터 (Data)

**SST (Stanford Sentiment Treebank)** — 영화 리뷰에서 추출한 단일 문장, 5단계 감정 레이블(0: very negative ~ 4: very positive). 모델은 마지막 토큰 임베딩으로 감정 레이블을 예측한다.

| split | 예제 수 |
|---|---|
| train | 8,544 |
| dev | 1,101 |
| test | 2,210 |

**CFIMDB (Compact Fine-grained IMDB)** — IMDB 영화 리뷰의 축소판으로, 극단적으로 양극화된 리뷰의 이진 분류(0: negative, 1: positive). 다수의 리뷰가 한 문장 이상의 길이를 가진다.

| split | 예제 수 |
|---|---|
| train | 1,701 |
| dev | 245 |
| test | 488 |

데이터셋은 모두 프로젝트 리포지토리의 `data/` 폴더에 `.csv` 형식으로 포함되어 있으며, 모델 학습·튜닝·평가에는 제공된 train/dev split만 사용하였다. test split은 최종 예측 산출에만 사용하고, 모델 개선에 활용하지 않았다(연구 윤리 준수).

### 4.2 평가 방법 (Evaluation)

두 데이터셋 모두 라벨 분포를 고려하여 **정확도(accuracy)**를 주 평가 지표로 사용하며, 클래스 불균형을 함께 점검하기 위해 **macro-F1**과 **혼동 행렬(confusion matrix)**을 보조 지표로 보고한다. 정량 비교는 라벨이 공개된 dev split에서 수행하였다.

### 4.3 실험 세부 정보 (Experimental Details)

| 항목 | last-linear-layer | full-model |
|---|---|---|
| 학습률 (lr) | 1e-3 | 1e-5 |
| Epochs | 10 | 10 |
| Batch size (SST / CFIMDB) | 64 / 8 | 64 / 8 |
| Hidden dropout | 0.3 | 0.3 |
| Optimizer | AdamW (직접 구현) | AdamW (직접 구현) |
| Random seed | 11711 | 11711 |
| 백본 | GPT-2 (117M, `gpt2`) | GPT-2 (117M, `gpt2`) |

학습 환경은 단일 NVIDIA GPU(CUDA)이며, 각 데이터셋 학습에는 GPU 사양에 따라 대략 5~15분이 소요되었다. 모든 명령행 옵션·하이퍼파라미터는 과제에서 제공한 기본값을 변경/추가 없이 사용하였고, PART-I에서는 `env.yml`에 포함된 패키지만 사용하였다.

### 4.4 결과 (Results)

dev split(SST 1,101개, CFIMDB 245개)에 대한 정량 결과는 다음과 같다. 네 가지 구성 모두 과제에서 제시한 기준 모델 정확도를 상회하였다.

| 모델 | Dev Accuracy | Macro-F1 | Baseline (Acc) | 차이 |
|---|---|---|---|---|
| SST / last-linear-layer | 0.4777 | 0.4180 | 0.462 | **+0.0157** |
| SST / full-model | 0.5159 | 0.4776 | 0.513 | **+0.0029** |
| CFIMDB / last-linear-layer | 0.8653 | 0.8647 | 0.861 | **+0.0043** |
| CFIMDB / full-model | 0.9796 | 0.9796 | 0.976 | **+0.0036** |

**혼동 행렬 (dev split, 행=정답 / 열=예측).**

CFIMDB / full-model (거의 완벽한 대각 행렬):

| | Pred Neg | Pred Pos |
|---|---|---|
| **True Neg** | 120 | 3 |
| **True Pos** | 2 | 120 |

SST / full-model (5-class, 오분류가 인접 등급에 집중):

| | P0 | P1 | P2 | P3 | P4 |
|---|---|---|---|---|---|
| **T0 (very neg)** | 38 | 85 | 11 | 5 | 0 |
| **T1 (neg)** | 21 | 197 | 42 | 29 | 0 |
| **T2 (neutral)** | 10 | 68 | 65 | 83 | 3 |
| **T3 (pos)** | 1 | 16 | 27 | 201 | 34 |
| **T4 (very pos)** | 1 | 1 | 6 | 90 | 67 |

**결과에 대한 논의.**

1. **기준 모델 대비.** 네 구성 모두 baseline을 넘었으나, SST·CFIMDB 모두 `full-model`의 초과 폭(+0.003~+0.004)이 `last-linear-layer`(+0.016, +0.004)보다 작다. 이는 baseline 자체가 full fine-tuning의 강한 성능을 기준으로 잡혀 있어 여유가 크지 않기 때문으로, seed에 따라 소폭 변동할 수 있다(향후 연구에서 다중 seed 검증 필요).

2. **full-model > last-linear-layer.** 두 데이터셋 모두 전체 미세조정이 우수했다(SST 0.516 vs 0.478, CFIMDB 0.980 vs 0.865). GPT-2를 동결하면 사전학습 표현을 그대로 쓰므로 감정 분류에 최적화되지 않은 반면, 전체를 갱신하면 표현 자체가 태스크에 적응한다. 특히 CFIMDB에서 격차(+0.114)가 큰데, 소규모(1,701개) 이진 데이터에서도 전체 미세조정이 표현을 효과적으로 재배치함을 보여준다.

3. **SST(5-class) ≪ CFIMDB(binary).** SST 정확도(0.52)가 CFIMDB(0.98)보다 크게 낮은 것은 (i) 클래스 수가 많고 (ii) 'somewhat negative/positive'와 'neutral'의 경계가 모호하기 때문이다. 혼동 행렬에서 오분류는 **인접 등급에 집중**(예: neutral(T2)의 다수가 P1·P3로, very positive(T4)의 다수가 P3로)되며, 극단 간 오분류(T4→P0 등)는 0에 가깝다. 즉 모델이 감정의 **순서(ordinal) 구조**는 포착하나 인접 등급 구분에서 한계를 보인다. neutral(T2)이 가장 어려워(full-model 기준 per-label F1 0.342) 전체 macro-F1을 끌어내린다.

4. **Accuracy vs Macro-F1.** CFIMDB는 클래스가 균형이라 accuracy ≈ macro-F1(0.9796)이다. 반면 SST는 macro-F1(0.478)이 accuracy(0.516)보다 낮은데, 이는 소수·경계 클래스(very negative, neutral)의 성능이 약해 모든 클래스를 동등 평균하면 점수가 떨어지기 때문이다. 따라서 다중·불균형 분류에서는 accuracy만으로 성능을 과대평가하지 않도록 macro-F1을 함께 보는 것이 타당하다.

---

## 5. 결론 및 향후 연구 (Conclusion & Future Work)

본 연구는 GPT-2의 핵심 구성 요소(Causal Self-Attention, Pre-LayerNorm 블록, 전체 모델)와 AdamW 옵티마이저를 PyTorch로 밑바닥부터 구현하고, 공식 구현 및 참조 가중치와의 수치 비교로 정합성을 검증하였다. 또한 구현한 모델을 SST와 CFIMDB에 미세조정하여, GPT-2를 동결한 `last-linear-layer` 방식과 전체를 갱신하는 `full-model` 방식을 비교하였다. 결과적으로 네 가지 구성 모두 기준 모델 정확도를 상회하였으며(SST full 0.516, CFIMDB full 0.980), 전체 미세조정이 두 데이터셋 모두에서 우수함을 확인하였다. SST의 오분류가 인접 감정 등급에 집중된다는 점에서 모델이 감정의 순서 구조를 포착함을 확인하였고, 동시에 중립(neutral) 등급 구분이 핵심 한계임을 혼동 행렬로 드러냈다.

향후 연구로는 (1) 학습률 스케줄링·warmup 도입을 통한 `full-model` 미세조정의 안정화, (2) 마지막 토큰 외에 평균 풀링(mean-pooling) 등 문장 표현 방식의 비교, (3) PART-II로의 확장(Paraphrase Detection의 cloze-style 재구성, Sonnet Generation)을 통해 동일 백본의 생성·판별 능력을 함께 평가하는 것을 고려할 수 있다.

---

## 6. 참고 문헌 (References)

[1] Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Y. Ng, and Christopher Potts. *Recursive deep models for semantic compositionality over a sentiment treebank.* In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1631–1642, 2013.

[2] Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. *Improving language understanding by generative pre-training.* 2018.

[3] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. *Language models are unsupervised multitask learners.* OpenAI blog, 1(8):9, 2019.

[4] Ilya Loshchilov and Frank Hutter. *Decoupled weight decay regularization.* In International Conference on Learning Representations (ICLR), 2019.

[5] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. *Attention is all you need.* In Advances in Neural Information Processing Systems (NeurIPS), 2017.

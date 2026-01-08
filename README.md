# 택배상자 질량 중심 제어 시스템 (PPO 강화학습)

> 자유낙하하는 택배상자가 바닥면으로 안정적으로 착지하도록 내부 질량체를 제어하는 강화학습 시스템

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [시스템 구조](#2-시스템-구조)
3. [설치 방법](#3-설치-방법)
4. [빠른 시작](#4-빠른-시작)
5. [MuJoCo 물리 모델 상세](#5-mujoco-물리-모델-상세)
6. [Gymnasium 환경 상세](#6-gymnasium-환경-상세)
7. [PPO 알고리즘 상세](#7-ppo-알고리즘-상세)
8. [훈련 가이드](#8-훈련-가이드)
9. [평가 가이드](#9-평가-가이드)
10. [하이퍼파라미터 튜닝](#10-하이퍼파라미터-튜닝)
11. [문제 해결](#11-문제-해결)

---

## 1. 프로젝트 개요

### 1.1 문제 정의

택배 배송 과정에서 상자가 떨어질 때, 모서리나 꼭지점으로 착지하면 내용물이 손상될 수 있습니다. 이 프로젝트는 **상자 내부의 이동 가능한 질량체를 제어**하여, 낙하 중 상자의 자세를 조정하고 **바닥면으로 안정적으로 착지**하도록 하는 시스템입니다.

### 1.2 접근 방식

| 구성 요소 | 기술 |
|-----------|------|
| 물리 시뮬레이션 | MuJoCo (DeepMind) |
| 강화학습 환경 | Gymnasium |
| 학습 알고리즘 | PPO (Proximal Policy Optimization) |
| 프레임워크 | Stable-Baselines3 |

### 1.3 시스템 사양

| 항목 | 사양 |
|------|------|
| 택배상자 크기 | 250 × 250 × 250 mm |
| 택배상자 무게 | 5 kg |
| 제어 질량체 무게 | 1 kg |
| 질량체 이동 범위 | ±100 mm (X, Y축) |
| 낙하 높이 | 3.7 m |
| 제어 주파수 | 50 Hz |
| 모터 최대 출력 | ±50 N |

### 1.4 프로젝트 구조

```
center_of_mass/
├── models/
│   └── delivery_box.xml          # MuJoCo 물리 모델 정의
├── envs/
│   ├── __init__.py
│   └── delivery_box_env.py       # Gymnasium 환경 클래스
├── scripts/
│   ├── __init__.py
│   ├── train_ppo.py              # PPO 훈련 스크립트
│   └── evaluate.py               # 모델 평가 스크립트
├── logs/                          # 훈련 로그 (TensorBoard)
├── requirements.txt               # 의존성 패키지
└── README.md                      # 이 문서
```

---

## 2. 시스템 구조

### 2.1 물리적 구조

```
┌─────────────────────────────────────┐
│           택배상자 (5kg)             │
│  ┌─────────────────────────────┐   │
│  │                             │   │
│  │     (빈 공간)                │   │
│  │                             │   │
│  ├─────────────────────────────┤   │ ← 상자 높이 250mm
│  │  ═══════════════════════    │   │ ← X축 레일
│  │          ●                  │   │ ← 질량체 (1kg)
│  │      ←─────→                │   │ ← Y축 이동
│  └─────────────────────────────┘   │
│         ↑                          │
│    X축 이동                         │
└─────────────────────────────────────┘
```

### 2.2 CoreXY 메커니즘

실제 CoreXY 시스템은 2개의 모터로 2축 이동을 구현합니다. 본 시뮬레이션에서는 이를 단순화하여 **2개의 독립적인 슬라이드 조인트**로 모델링했습니다.

```
모터 X → joint_x → X축 이동 (±100mm)
모터 Y → joint_y → Y축 이동 (±100mm)
```

### 2.3 강화학습 구조

```
┌──────────────────────────────────────────────────────────────┐
│                      강화학습 루프                            │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│   ┌─────────┐    관측 (19차원)    ┌─────────────────┐        │
│   │         │ ──────────────────→ │                 │        │
│   │  환경   │                     │   PPO 에이전트   │        │
│   │(MuJoCo) │ ←────────────────── │  (신경망 정책)   │        │
│   │         │    행동 (2차원)     │                 │        │
│   └─────────┘                     └─────────────────┘        │
│        │                                   │                  │
│        │ 보상                              │ 학습             │
│        └───────────────────────────────────┘                  │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. 설치 방법

### 3.1 요구 사항

- Python 3.10 이상
- CUDA (선택, GPU 가속용)

### 3.2 설치

```bash
# 저장소 클론
git clone https://github.com/your-repo/center_of_mass.git
cd center_of_mass

# 가상환경 생성 (권장)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 3.3 의존성 패키지

```
gymnasium>=0.29.0       # 강화학습 환경 인터페이스
stable-baselines3>=2.2.0 # PPO 알고리즘 구현
mujoco>=3.0.0           # 물리 시뮬레이션 엔진
numpy>=1.24.0           # 수치 계산
matplotlib>=3.7.0       # 시각화 (선택)
tensorboard>=2.14.0     # 학습 모니터링 (선택)
```

---

## 4. 빠른 시작

### 4.1 환경 테스트 (랜덤 에이전트)

```bash
python scripts/evaluate.py --test-random --n-episodes 5
```

예상 출력:
```
======================================================================
랜덤 에이전트 테스트
======================================================================
  에피소드   1: 보상=  12.45, 기울기= 35.67°, 성공=X
  에피소드   2: 보상=  45.23, 기울기= 12.34°, 성공=O
  ...
----------------------------------------------------------------------
랜덤 에이전트 성공률: 20.0%
평균 착지 기울기: 28.45°
```

### 4.2 모델 훈련

```bash
# 기본 훈련 (100만 스텝)
python scripts/train_ppo.py

# 모든 기능 활성화
python scripts/train_ppo.py \
    --total-timesteps 2000000 \
    --curriculum \
    --domain-rand \
    --vec-normalize
```

### 4.3 학습된 모델 평가

```bash
python scripts/evaluate.py \
    --model models/trained/ppo_delivery_box_YYYYMMDD_HHMMSS/final_model.zip \
    --n-episodes 100
```

---

## 5. MuJoCo 물리 모델 상세

### 5.1 파일 위치

`models/delivery_box.xml`

### 5.2 좌표계 및 단위

| 물리량 | 단위 |
|--------|------|
| 길이 | 미터 (m) |
| 질량 | 킬로그램 (kg) |
| 시간 | 초 (s) |
| 힘 | 뉴턴 (N) |
| 각도 | 라디안 (rad) |

### 5.3 시뮬레이션 설정

```xml
<option timestep="0.001" gravity="0 0 -9.81" integrator="RK4"/>
```

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `timestep` | 0.001 | 1ms마다 물리 계산 (1000Hz) |
| `gravity` | 0 0 -9.81 | Z축 아래로 9.81 m/s² |
| `integrator` | RK4 | 4차 룽게-쿠타 (높은 정확도) |

### 5.4 바디 계층 구조

```
worldbody
├── floor (바닥, 충돌 활성화)
│
└── box_body (택배상자, freejoint)
    ├── box_bottom (바닥면, 1.0kg)
    ├── box_top (윗면, 0.8kg)
    ├── box_front (앞면, 0.8kg)
    ├── box_back (뒷면, 0.8kg)
    ├── box_left (왼쪽면, 0.8kg)
    ├── box_right (오른쪽면, 0.8kg)
    │
    └── x_rail (X축 레일, slide joint)
        └── mass_body (질량체, slide joint)
            └── control_mass (1.0kg, 원통형)
```

### 5.5 조인트 상세

#### freejoint (box_joint)
- **자유도**: 6 (위치 3 + 회전 3)
- **용도**: 상자가 공중에서 자유롭게 움직이도록 함
- **qpos 구조**: [x, y, z, qw, qx, qy, qz] (7개)
- **qvel 구조**: [vx, vy, vz, ωx, ωy, ωz] (6개)

#### slide joint (joint_x, joint_y)
- **자유도**: 1 (직선 운동)
- **이동 범위**: ±0.10 m (±100mm)
- **감쇠**: 0.5 Ns/m

### 5.6 액추에이터 (모터)

```xml
<motor name="motor_x" joint="joint_x" gear="1" ctrlrange="-50 50"/>
<motor name="motor_y" joint="joint_y" gear="1" ctrlrange="-50 50"/>
```

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `gear` | 1 | 기어비 1:1 |
| `ctrlrange` | -50 ~ 50 | 최대 힘 ±50N |

### 5.7 충돌 설정

| 객체 | contype | conaffinity | 충돌 대상 |
|------|---------|-------------|-----------|
| floor | 1 | 1 | 상자 바닥면 |
| box_* (6면) | 1 | 1 | 바닥 |
| x_rail | 0 | 0 | 없음 |
| control_mass | 0 | 0 | 없음 |

---

## 6. Gymnasium 환경 상세

### 6.1 파일 위치

`envs/delivery_box_env.py`

### 6.2 환경 생성

```python
from envs.delivery_box_env import DeliveryBoxEnv, DomainRandomization, CurriculumConfig

# 기본 환경
env = DeliveryBoxEnv()

# 커스텀 설정
env = DeliveryBoxEnv(
    xml_path="/path/to/model.xml",
    drop_height=3.7,
    max_episode_steps=2000,
    control_frequency=50,
    domain_randomization=DomainRandomization(enabled=True),
    curriculum=CurriculumConfig(enabled=True),
)
```

### 6.3 관측 공간 (Observation Space)

**차원**: 19

| 인덱스 | 이름 | 차원 | 범위 | 설명 |
|--------|------|------|------|------|
| 0-2 | box_pos | 3 | ~4m | 상자 월드 좌표 위치 |
| 3-6 | box_quat | 4 | -1~1 | 상자 자세 (쿼터니언) |
| 7-9 | box_linvel | 3 | ~10m/s | 상자 선속도 |
| 10-12 | box_angvel | 3 | ~10rad/s | 상자 각속도 |
| 13-14 | mass_pos | 2 | -0.1~0.1 | 질량체 위치 (m) |
| 15-16 | mass_vel | 2 | ~1m/s | 질량체 속도 |
| 17 | time_to_landing | 1 | 0~1 | 착지 예상 시간 (정규화) |
| 18 | normalized_height | 1 | 0~1 | 정규화된 높이 |

### 6.4 행동 공간 (Action Space)

**차원**: 2
**범위**: [-1, 1]

| 인덱스 | 이름 | 설명 |
|--------|------|------|
| 0 | motor_x | X축 모터 제어 (-1=후진, +1=전진) |
| 1 | motor_y | Y축 모터 제어 (-1=후진, +1=전진) |

실제 힘 = action × 50N × actuator_scale

### 6.5 보상 함수 (Reward Function)

#### 낙하 중 보상 (height > 0.2m)

```python
# 1. 수평 유지 보상 (연속형)
tilt_reward = exp(-5.0 × tilt_rad)
reward += 0.1 × tilt_reward

# 2. 제어 비용 패널티
control_cost = 0.001 × (action[0]² + action[1]²)
reward -= control_cost

# 3. 제어 변화량 패널티
action_diff = (action - prev_action)²
reward -= 0.0005 × action_diff
```

#### 착지 보상 (height < 0.15m)

```python
# 1. 기울기 보상 (연속형)
tilt_reward = 100.0 × exp(-5.0 × tilt_rad)

# 2. 충격 패널티
expected_v = sqrt(2 × 9.81 × drop_height)  # ≈ 8.5 m/s
impact_penalty = 10.0 × (landing_vz / expected_v)

# 3. 성공 보너스
if tilt_degrees < 5:
    success_bonus = 20.0
elif tilt_degrees < 15:
    success_bonus = 10.0
else:
    success_bonus = 0.0

# 4. 실패 패널티
if tilt_degrees >= 45:
    failure_penalty = 30.0
else:
    failure_penalty = 0.0

# 최종 착지 보상
landing_reward = tilt_reward - impact_penalty + success_bonus - failure_penalty
```

#### 보상 테이블

| 착지 기울기 | tilt_reward | impact | bonus | penalty | 총 보상 |
|-------------|-------------|--------|-------|---------|---------|
| 0° | 100.0 | -10.0 | +20.0 | 0 | **+110** |
| 5° | 64.1 | -10.0 | +20.0 | 0 | **+74** |
| 10° | 41.1 | -10.0 | +10.0 | 0 | **+41** |
| 15° | 26.4 | -10.0 | 0 | 0 | **+16** |
| 30° | 7.0 | -10.0 | 0 | 0 | **-3** |
| 45° | 1.8 | -10.0 | 0 | -30.0 | **-38** |

### 6.6 에피소드 종료 조건

| 조건 | terminated | truncated |
|------|------------|-----------|
| 착지 (height < 0.15m) | True | False |
| 시간 초과 (2000 스텝) | False | True |
| 비정상 (height < -0.5 or > 15) | True | False |

### 6.7 도메인 랜덤화

```python
DomainRandomization(
    enabled=True,
    mass_range=(0.9, 1.1),           # 질량 ±10%
    friction_range=(0.8, 1.2),       # 마찰 ±20%
    actuator_scale_range=(0.9, 1.1), # 모터 출력 ±10%
    sensor_noise_std=0.01,           # 센서 노이즈 1cm
)
```

### 6.8 커리큘럼 학습

```python
CurriculumConfig(
    enabled=True,
    initial_drop_height=1.0,    # 시작 높이: 1m
    final_drop_height=3.7,      # 최종 높이: 3.7m
    initial_tilt_range=5.0,     # 시작 기울기: ±5°
    final_tilt_range=15.0,      # 최종 기울기: ±15°
)
```

| 진행률 | 낙하 높이 | 초기 기울기 |
|--------|-----------|-------------|
| 0% | 1.0m | ±5° |
| 25% | 1.675m | ±7.5° |
| 50% | 2.35m | ±10° |
| 75% | 3.025m | ±12.5° |
| 100% | 3.7m | ±15° |

---

## 7. PPO 알고리즘 상세

### 7.1 PPO (Proximal Policy Optimization) 개요

PPO는 정책 그래디언트 기반의 강화학습 알고리즘으로, **정책 변화 폭을 제한**하여 안정적인 학습을 보장합니다.

### 7.2 핵심 수식

#### 클리핑 목적 함수

```
L_CLIP(θ) = E[min(r(θ)Â, clip(r(θ), 1-ε, 1+ε)Â)]

r(θ) = π_θ(a|s) / π_θ_old(a|s)  (확률 비율)
ε = 0.2 (클리핑 범위)
Â = 어드밴티지 추정치
```

#### Generalized Advantage Estimation (GAE)

```
Â_t = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}
δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

### 7.3 하이퍼파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| learning_rate | 3e-4 | Adam 학습률 |
| n_steps | 2048 | 환경당 수집 스텝 수 |
| batch_size | 64 | 미니배치 크기 |
| n_epochs | 10 | 업데이트 반복 횟수 |
| gamma | 0.99 | 할인율 |
| gae_lambda | 0.95 | GAE 람다 |
| clip_range | 0.2 | PPO 클리핑 범위 |
| ent_coef | 0.01 | 엔트로피 계수 |
| vf_coef | 0.5 | 가치 함수 손실 가중치 |
| max_grad_norm | 0.5 | 그래디언트 클리핑 |

### 7.4 신경망 구조

```
정책 네트워크 (Actor):
  입력 (19) → Dense(256) → ReLU → Dense(256) → ReLU → Dense(2) → Tanh

가치 네트워크 (Critic):
  입력 (19) → Dense(256) → ReLU → Dense(256) → ReLU → Dense(1)
```

### 7.5 학습 루프

```
for iteration in range(total_iterations):
    # 1. 경험 수집
    for step in range(n_steps):
        action = policy.sample(obs)
        next_obs, reward, done, info = env.step(action)
        buffer.add(obs, action, reward, done)
        obs = next_obs

    # 2. GAE 계산
    advantages = compute_gae(buffer, value_network)
    returns = advantages + values

    # 3. PPO 업데이트
    for epoch in range(n_epochs):
        for batch in buffer.sample(batch_size):
            # 정책 손실
            ratio = new_prob / old_prob
            clipped_ratio = clip(ratio, 1-ε, 1+ε)
            policy_loss = -min(ratio * adv, clipped_ratio * adv)

            # 가치 손실
            value_loss = (value - returns)²

            # 엔트로피 보너스
            entropy = -sum(prob * log(prob))

            # 총 손실
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            # 역전파
            optimizer.step(loss)
```

---

## 8. 훈련 가이드

### 8.1 기본 훈련

```bash
python scripts/train_ppo.py
```

### 8.2 명령줄 옵션

#### 훈련 설정

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--total-timesteps` | 1,000,000 | 총 훈련 스텝 수 |
| `--n-envs` | 4 | 병렬 환경 수 |
| `--seed` | 42 | 랜덤 시드 |
| `--device` | auto | 학습 디바이스 (cpu/cuda) |

#### 개선 기능

| 옵션 | 설명 |
|------|------|
| `--curriculum` | 커리큘럼 학습 활성화 |
| `--domain-rand` | 도메인 랜덤화 활성화 |
| `--vec-normalize` | 관측/보상 정규화 활성화 |
| `--lr-schedule` | 선형 학습률 감소 활성화 |

#### PPO 하이퍼파라미터

| 옵션 | 기본값 |
|------|--------|
| `--learning-rate` | 3e-4 |
| `--n-steps` | 2048 |
| `--batch-size` | 64 |
| `--n-epochs` | 10 |
| `--gamma` | 0.99 |
| `--gae-lambda` | 0.95 |
| `--clip-range` | 0.2 |
| `--ent-coef` | 0.01 |
| `--vf-coef` | 0.5 |

### 8.3 권장 훈련 설정

#### 빠른 테스트
```bash
python scripts/train_ppo.py --total-timesteps 100000 --n-envs 4
```

#### 기본 훈련
```bash
python scripts/train_ppo.py --total-timesteps 1000000 --n-envs 8
```

#### 최적 훈련 (권장)
```bash
python scripts/train_ppo.py \
    --total-timesteps 2000000 \
    --n-envs 8 \
    --curriculum \
    --domain-rand \
    --vec-normalize \
    --lr-schedule
```

### 8.4 TensorBoard 모니터링

```bash
tensorboard --logdir logs/
```

브라우저에서 `http://localhost:6006` 접속

### 8.5 출력 파일

훈련 완료 후 생성되는 파일:

```
models/trained/ppo_delivery_box_YYYYMMDD_HHMMSS_seedXX/
├── final_model.zip           # 최종 모델
├── best/
│   └── best_model.zip        # 최고 성능 모델
├── checkpoints/
│   ├── ppo_delivery_box_50000_steps.zip
│   ├── ppo_delivery_box_100000_steps.zip
│   └── ...
├── vec_normalize.pkl         # 정규화 통계 (--vec-normalize 사용 시)
└── experiment_config.json    # 실험 설정
```

---

## 9. 평가 가이드

### 9.1 기본 평가

```bash
python scripts/evaluate.py \
    --model models/trained/.../final_model.zip \
    --n-episodes 100
```

### 9.2 명령줄 옵션

| 옵션 | 설명 |
|------|------|
| `--model` | 모델 파일 경로 (.zip) |
| `--vec-normalize` | VecNormalize 파일 경로 (.pkl) |
| `--n-episodes` | 평가 에피소드 수 (기본: 100) |
| `--seed` | 랜덤 시드 |
| `--stochastic` | 확률적 행동 사용 (기본: 결정적) |
| `--multi-seed` | 다중 시드 평가 (5개 시드) |
| `--save` | 결과 저장 경로 (JSON) |
| `--test-random` | 랜덤 에이전트 테스트 |

### 9.3 평가 결과 해석

```
======================================================================
평가 결과 요약
======================================================================

성공률:                  85.0%

[ 보상 통계 ]
  평균 ± 표준편차:       78.45 ± 23.12
  범위:                  [-15.23, 112.34]

[ 착지 기울기 (도) ]
  평균 ± 표준편차:       8.45 ± 5.23
  중앙값:                7.12
  90 백분위수:           15.67
  범위:                  [0.45, 42.34]

[ 착지 충격 ]
  충격 패널티:           6.32 ± 1.45
  평균 착지 속도 (Z):    8.12 m/s

[ 제어 비용 ]
  누적 제어 비용:        0.0234 ± 0.0123

[ 에피소드 길이 ]
  평균 스텝:             45.2 ± 12.3
======================================================================
```

### 9.4 다중 시드 평가

재현성 검증을 위한 다중 시드 평가:

```bash
python scripts/evaluate.py \
    --model models/trained/.../final_model.zip \
    --multi-seed \
    --save results.json
```

출력:
```
======================================================================
다중 시드 통합 결과
======================================================================
성공률: 84.2% ± 3.1%
평균 기울기: 8.67° ± 1.23°
```

---

## 10. 하이퍼파라미터 튜닝

### 10.1 학습률 (learning_rate)

| 값 | 효과 |
|----|------|
| 1e-2 | 불안정, 발산 위험 |
| 3e-4 | 권장값 (논문 기본) |
| 1e-5 | 너무 느림 |

### 10.2 배치 크기 (batch_size)

| 값 | 효과 |
|----|------|
| 32 | 높은 분산, 빠른 업데이트 |
| 64 | 권장값 |
| 256 | 안정적, 메모리 많이 사용 |

### 10.3 엔트로피 계수 (ent_coef)

| 값 | 효과 |
|----|------|
| 0.001 | 적은 탐험 |
| 0.01 | 권장값 |
| 0.1 | 과도한 탐험 |

### 10.4 클리핑 범위 (clip_range)

| 값 | 효과 |
|----|------|
| 0.1 | 보수적 업데이트 |
| 0.2 | 권장값 |
| 0.3 | 공격적 업데이트 |

### 10.5 튜닝 전략

1. **기본값으로 시작**: 위의 권장값 사용
2. **학습 곡선 확인**: TensorBoard로 모니터링
3. **문제 진단**:
   - 수렴 안 함 → 학습률 조정
   - 불안정 → 클리핑 범위 축소
   - 탐험 부족 → 엔트로피 계수 증가
4. **그리드 서치**: 여러 조합 실험

---

## 11. 문제 해결

### 11.1 설치 관련

#### MuJoCo 설치 오류
```bash
# Linux
pip install mujoco

# 라이브러리 누락 시
sudo apt-get install libgl1-mesa-glx libosmesa6
```

#### CUDA 관련 오류
```bash
# CPU만 사용
python scripts/train_ppo.py --device cpu
```

### 11.2 훈련 관련

#### 학습이 진행되지 않음
- 학습률 확인 (너무 작거나 큼)
- 보상 스케일 확인
- `--vec-normalize` 시도

#### 불안정한 학습
- `--lr-schedule` 사용
- `clip_range` 감소 (0.1)
- `max_grad_norm` 감소 (0.3)

#### 메모리 부족
- `--n-envs` 감소
- `--batch-size` 감소

### 11.3 평가 관련

#### 모델 로드 오류
```bash
# 경로에 공백이 있으면 따옴표 사용
python scripts/evaluate.py --model "path with spaces/model.zip"
```

#### VecNormalize 불일치
```bash
# 훈련 시 --vec-normalize 사용했으면 평가 시에도 필요
python scripts/evaluate.py \
    --model model.zip \
    --vec-normalize vec_normalize.pkl
```

### 11.4 흔한 오류 메시지

| 오류 | 원인 | 해결 |
|------|------|------|
| `mass and inertia must be larger than mjMINVAL` | 바디에 질량 없음 | XML에 geom 추가 |
| `unrecognized attribute` | MuJoCo 버전 차이 | 속성 제거 또는 버전 업그레이드 |
| `FileNotFoundError: model.zip` | 경로 오류 | 정확한 경로 확인 |

---

## 참고 자료

- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [PPO Paper (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)

---

## 라이선스

MIT License

---

## 기여

이슈 및 풀 리퀘스트 환영합니다!

#!/usr/bin/env python3
"""
PPO 강화학습 훈련 스크립트

택배상자 질량 중심 제어를 위한 PPO 에이전트 훈련
- 알고리즘: PPO (Proximal Policy Optimization)
- 프레임워크: Stable-Baselines3 + Gymnasium
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

# 프로젝트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from envs.delivery_box_env import DeliveryBoxEnv


def make_env(rank: int, seed: int = 0):
    """
    환경 생성 함수 (멀티프로세싱용)
    """
    def _init():
        env = DeliveryBoxEnv()
        env.reset(seed=seed + rank)
        return Monitor(env)
    set_random_seed(seed)
    return _init


def linear_schedule(initial_value: float):
    """
    선형 학습률 스케줄
    학습 진행에 따라 학습률을 선형적으로 감소
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def train(args):
    """
    PPO 에이전트 훈련
    """
    # 실험 디렉토리 설정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"ppo_delivery_box_{timestamp}"
    log_dir = project_root / "logs" / experiment_name
    model_dir = project_root / "models" / "trained" / experiment_name

    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("택배상자 질량 중심 제어 - PPO 강화학습 훈련")
    print("=" * 60)
    print(f"실험 이름: {experiment_name}")
    print(f"로그 디렉토리: {log_dir}")
    print(f"모델 디렉토리: {model_dir}")
    print()

    # 환경 생성
    print("[1/4] 환경 생성 중...")
    if args.n_envs > 1:
        # 멀티프로세싱 환경
        env = SubprocVecEnv([make_env(i, args.seed) for i in range(args.n_envs)])
    else:
        # 단일 환경
        env = DummyVecEnv([make_env(0, args.seed)])

    # 평가 환경
    eval_env = DummyVecEnv([make_env(0, args.seed + 1000)])

    print(f"  - 훈련 환경 수: {args.n_envs}")
    print(f"  - 관측 공간: {env.observation_space}")
    print(f"  - 행동 공간: {env.action_space}")
    print()

    # PPO 하이퍼파라미터 설정
    print("[2/4] PPO 에이전트 설정 중...")

    # 학습률 스케줄
    if args.lr_schedule:
        learning_rate = linear_schedule(args.learning_rate)
    else:
        learning_rate = args.learning_rate

    # PPO 에이전트 생성
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=args.target_kl,
        tensorboard_log=str(log_dir),
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
        ),
        verbose=1,
        seed=args.seed,
        device=args.device,
    )

    print("  PPO 하이퍼파라미터:")
    print(f"    - 학습률: {args.learning_rate}")
    print(f"    - n_steps: {args.n_steps}")
    print(f"    - batch_size: {args.batch_size}")
    print(f"    - n_epochs: {args.n_epochs}")
    print(f"    - gamma: {args.gamma}")
    print(f"    - gae_lambda: {args.gae_lambda}")
    print(f"    - clip_range: {args.clip_range}")
    print(f"    - ent_coef: {args.ent_coef}")
    print(f"    - vf_coef: {args.vf_coef}")
    print()

    # 콜백 설정
    print("[3/4] 콜백 설정 중...")

    # 평가 콜백
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / "best"),
        log_path=str(log_dir / "eval"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )

    # 체크포인트 콜백
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(model_dir / "checkpoints"),
        name_prefix="ppo_delivery_box",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    callbacks = CallbackList([eval_callback, checkpoint_callback])
    print(f"  - 평가 주기: {args.eval_freq} steps")
    print(f"  - 체크포인트 주기: {args.checkpoint_freq} steps")
    print()

    # 훈련 시작
    print("[4/4] 훈련 시작!")
    print(f"  - 총 타임스텝: {args.total_timesteps:,}")
    print("-" * 60)

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            log_interval=args.log_interval,
            progress_bar=False,
        )
    except KeyboardInterrupt:
        print("\n훈련이 사용자에 의해 중단되었습니다.")

    # 최종 모델 저장
    final_model_path = model_dir / "final_model"
    model.save(str(final_model_path))
    print()
    print("-" * 60)
    print(f"최종 모델 저장됨: {final_model_path}.zip")
    print("훈련 완료!")

    # 환경 정리
    env.close()
    eval_env.close()

    return model, str(model_dir)


def main():
    parser = argparse.ArgumentParser(
        description="택배상자 질량 중심 제어 PPO 훈련"
    )

    # 훈련 설정
    parser.add_argument(
        "--total-timesteps", type=int, default=1_000_000,
        help="총 훈련 타임스텝 (기본값: 1,000,000)"
    )
    parser.add_argument(
        "--n-envs", type=int, default=4,
        help="병렬 환경 수 (기본값: 4)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="랜덤 시드 (기본값: 42)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"],
        help="학습 디바이스 (기본값: auto)"
    )

    # PPO 하이퍼파라미터
    parser.add_argument(
        "--learning-rate", type=float, default=3e-4,
        help="학습률 (기본값: 3e-4)"
    )
    parser.add_argument(
        "--lr-schedule", action="store_true",
        help="선형 학습률 스케줄 사용"
    )
    parser.add_argument(
        "--n-steps", type=int, default=2048,
        help="환경당 스텝 수 (기본값: 2048)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="미니배치 크기 (기본값: 64)"
    )
    parser.add_argument(
        "--n-epochs", type=int, default=10,
        help="PPO 업데이트 에폭 수 (기본값: 10)"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99,
        help="할인율 (기본값: 0.99)"
    )
    parser.add_argument(
        "--gae-lambda", type=float, default=0.95,
        help="GAE lambda (기본값: 0.95)"
    )
    parser.add_argument(
        "--clip-range", type=float, default=0.2,
        help="PPO 클리핑 범위 (기본값: 0.2)"
    )
    parser.add_argument(
        "--ent-coef", type=float, default=0.01,
        help="엔트로피 계수 (기본값: 0.01)"
    )
    parser.add_argument(
        "--vf-coef", type=float, default=0.5,
        help="가치 함수 계수 (기본값: 0.5)"
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=0.5,
        help="그래디언트 클리핑 (기본값: 0.5)"
    )
    parser.add_argument(
        "--target-kl", type=float, default=None,
        help="타겟 KL divergence (기본값: None)"
    )

    # 로깅/저장 설정
    parser.add_argument(
        "--log-interval", type=int, default=10,
        help="로그 출력 간격 (기본값: 10)"
    )
    parser.add_argument(
        "--eval-freq", type=int, default=10000,
        help="평가 주기 (기본값: 10000)"
    )
    parser.add_argument(
        "--n-eval-episodes", type=int, default=10,
        help="평가 에피소드 수 (기본값: 10)"
    )
    parser.add_argument(
        "--checkpoint-freq", type=int, default=50000,
        help="체크포인트 저장 주기 (기본값: 50000)"
    )

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

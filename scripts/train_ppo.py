#!/usr/bin/env python3
"""
PPO 강화학습 훈련 스크립트 (개선 버전)

개선 사항:
1. VecNormalize 적용 (관측/보상 정규화)
2. 커리큘럼 학습 콜백
3. 실험 설정 JSON 저장
4. 다중 시드 실험 지원
5. 도메인 랜덤화 옵션
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

# 프로젝트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from envs.delivery_box_env import DeliveryBoxEnv, DomainRandomization, CurriculumConfig


class CurriculumCallback(BaseCallback):
    """
    커리큘럼 학습 콜백

    학습 진행에 따라 환경 난이도를 점진적으로 증가
    """

    def __init__(
        self,
        total_timesteps: int,
        start_progress: float = 0.0,
        end_progress: float = 1.0,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.start_progress = start_progress
        self.end_progress = end_progress

    def _on_step(self) -> bool:
        # 현재 진행률 계산
        progress = self.num_timesteps / self.total_timesteps
        curriculum_progress = (
            self.start_progress +
            progress * (self.end_progress - self.start_progress)
        )

        # 모든 환경에 커리큘럼 진행률 설정
        for env in self.training_env.envs:
            if hasattr(env, 'env'):
                # Monitor 래퍼를 통과
                actual_env = env.env
                if hasattr(actual_env, 'set_curriculum_progress'):
                    actual_env.set_curriculum_progress(curriculum_progress)

        if self.verbose > 0 and self.num_timesteps % 50000 == 0:
            print(f"  [Curriculum] Progress: {curriculum_progress:.2%}")

        return True


class MetricsCallback(BaseCallback):
    """
    상세 메트릭 수집 콜백
    """

    def __init__(self, log_path: Path, verbose: int = 0):
        super().__init__(verbose)
        self.log_path = log_path
        self.episode_metrics = []

    def _on_step(self) -> bool:
        # 에피소드 종료 시 메트릭 수집
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                metrics = {
                    'timestep': self.num_timesteps,
                    'reward': info['episode']['r'],
                    'length': info['episode']['l'],
                }

                # 착지 정보가 있으면 추가
                if 'landing_tilt_degrees' in info:
                    metrics.update({
                        'landing_tilt': info['landing_tilt_degrees'],
                        'landing_velocity': info.get('landing_velocity', 0),
                        'success': info.get('success', False),
                        'impact_penalty': info.get('impact_penalty', 0),
                    })

                self.episode_metrics.append(metrics)

        return True

    def _on_training_end(self) -> None:
        # 메트릭 저장
        metrics_file = self.log_path / 'episode_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.episode_metrics, f, indent=2)


def make_env(
    rank: int,
    seed: int = 0,
    xml_path: Optional[str] = None,
    domain_rand: Optional[DomainRandomization] = None,
    curriculum: Optional[CurriculumConfig] = None,
):
    """환경 생성 함수"""
    def _init():
        env = DeliveryBoxEnv(
            xml_path=xml_path,
            domain_randomization=domain_rand,
            curriculum=curriculum,
        )
        env.reset(seed=seed + rank)
        return Monitor(env)
    set_random_seed(seed)
    return _init


def save_experiment_config(args: argparse.Namespace, save_path: Path):
    """실험 설정을 JSON으로 저장"""
    config = {
        'timestamp': datetime.now().isoformat(),
        'args': vars(args),
        'environment': {
            'drop_height': 3.7,
            'control_frequency': 50,
            'max_episode_steps': 2000,
        },
        'ppo_hyperparameters': {
            'learning_rate': args.learning_rate,
            'n_steps': args.n_steps,
            'batch_size': args.batch_size,
            'n_epochs': args.n_epochs,
            'gamma': args.gamma,
            'gae_lambda': args.gae_lambda,
            'clip_range': args.clip_range,
            'ent_coef': args.ent_coef,
            'vf_coef': args.vf_coef,
            'max_grad_norm': args.max_grad_norm,
        },
        'training': {
            'total_timesteps': args.total_timesteps,
            'n_envs': args.n_envs,
            'seed': args.seed,
            'use_curriculum': args.curriculum,
            'use_domain_rand': args.domain_rand,
            'use_vec_normalize': args.vec_normalize,
        }
    }

    config_file = save_path / 'experiment_config.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"  실험 설정 저장됨: {config_file}")


def linear_schedule(initial_value: float):
    """선형 학습률 스케줄"""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def train(args):
    """PPO 에이전트 훈련"""
    # 실험 디렉토리 설정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"ppo_delivery_box_{timestamp}_seed{args.seed}"
    log_dir = project_root / "logs" / experiment_name
    model_dir = project_root / "models" / "trained" / experiment_name

    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("택배상자 질량 중심 제어 - PPO 강화학습 훈련 (개선 버전)")
    print("=" * 70)
    print(f"실험 이름: {experiment_name}")
    print(f"로그 디렉토리: {log_dir}")
    print(f"모델 디렉토리: {model_dir}")
    print()

    # 실험 설정 저장
    save_experiment_config(args, model_dir)

    # 도메인 랜덤화 설정
    domain_rand = None
    if args.domain_rand:
        domain_rand = DomainRandomization(
            enabled=True,
            actuator_scale_range=(0.9, 1.1),
            sensor_noise_std=0.01,
        )
        print("  [Domain Randomization] 활성화")

    # 커리큘럼 설정
    curriculum = None
    if args.curriculum:
        curriculum = CurriculumConfig(
            enabled=True,
            initial_drop_height=1.0,
            final_drop_height=3.7,
            initial_tilt_range=5.0,
            final_tilt_range=15.0,
        )
        print("  [Curriculum Learning] 활성화")

    # 환경 생성
    print("\n[1/5] 환경 생성 중...")
    env_kwargs = {
        'domain_rand': domain_rand,
        'curriculum': curriculum,
    }

    if args.n_envs > 1:
        env = SubprocVecEnv([
            make_env(i, args.seed, **env_kwargs)
            for i in range(args.n_envs)
        ])
    else:
        env = DummyVecEnv([make_env(0, args.seed, **env_kwargs)])

    # VecNormalize 적용
    if args.vec_normalize:
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=args.gamma,
        )
        print("  [VecNormalize] 관측/보상 정규화 활성화")

    # 평가 환경 (정규화 없이)
    eval_env = DummyVecEnv([make_env(0, args.seed + 1000)])
    if args.vec_normalize:
        eval_env = VecNormalize(
            eval_env,
            norm_obs=True,
            norm_reward=False,  # 평가 시 보상은 정규화하지 않음
            clip_obs=10.0,
            training=False,
        )

    print(f"  - 훈련 환경 수: {args.n_envs}")
    print(f"  - 관측 공간: {env.observation_space}")
    print(f"  - 행동 공간: {env.action_space}")

    # PPO 하이퍼파라미터 설정
    print("\n[2/5] PPO 에이전트 설정 중...")

    learning_rate = linear_schedule(args.learning_rate) if args.lr_schedule else args.learning_rate

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
    print(f"    - 학습률: {args.learning_rate} {'(선형 감소)' if args.lr_schedule else ''}")
    print(f"    - n_steps: {args.n_steps}")
    print(f"    - batch_size: {args.batch_size}")
    print(f"    - n_epochs: {args.n_epochs}")
    print(f"    - gamma: {args.gamma}")
    print(f"    - clip_range: {args.clip_range}")
    print(f"    - ent_coef: {args.ent_coef}")

    # 콜백 설정
    print("\n[3/5] 콜백 설정 중...")

    callbacks = []

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
    callbacks.append(eval_callback)

    # 체크포인트 콜백
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(model_dir / "checkpoints"),
        name_prefix="ppo_delivery_box",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)

    # 커리큘럼 콜백
    if args.curriculum:
        curriculum_callback = CurriculumCallback(
            total_timesteps=args.total_timesteps,
            verbose=1,
        )
        callbacks.append(curriculum_callback)

    # 메트릭 콜백
    metrics_callback = MetricsCallback(log_path=log_dir, verbose=0)
    callbacks.append(metrics_callback)

    callback = CallbackList(callbacks)

    print(f"  - 평가 주기: {args.eval_freq:,} steps")
    print(f"  - 체크포인트 주기: {args.checkpoint_freq:,} steps")

    # 훈련 시작
    print("\n[4/5] 훈련 시작!")
    print(f"  - 총 타임스텝: {args.total_timesteps:,}")
    print("-" * 70)

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            log_interval=args.log_interval,
            progress_bar=False,
        )
    except KeyboardInterrupt:
        print("\n훈련이 사용자에 의해 중단되었습니다.")

    # 모델 저장
    print("\n[5/5] 모델 저장 중...")

    final_model_path = model_dir / "final_model"
    model.save(str(final_model_path))

    # VecNormalize 통계 저장
    if args.vec_normalize:
        vec_normalize_path = model_dir / "vec_normalize.pkl"
        env.save(str(vec_normalize_path))
        print(f"  VecNormalize 저장됨: {vec_normalize_path}")

    print()
    print("=" * 70)
    print(f"최종 모델 저장됨: {final_model_path}.zip")
    print("훈련 완료!")
    print("=" * 70)

    # 환경 정리
    env.close()
    eval_env.close()

    return model, str(model_dir)


def main():
    parser = argparse.ArgumentParser(
        description="택배상자 질량 중심 제어 PPO 훈련 (개선 버전)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 훈련 설정
    parser.add_argument(
        "--total-timesteps", type=int, default=1_000_000,
        help="총 훈련 타임스텝"
    )
    parser.add_argument(
        "--n-envs", type=int, default=4,
        help="병렬 환경 수"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="랜덤 시드"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"],
        help="학습 디바이스"
    )

    # 개선된 기능
    parser.add_argument(
        "--curriculum", action="store_true",
        help="커리큘럼 학습 활성화"
    )
    parser.add_argument(
        "--domain-rand", action="store_true",
        help="도메인 랜덤화 활성화"
    )
    parser.add_argument(
        "--vec-normalize", action="store_true",
        help="VecNormalize (관측/보상 정규화) 활성화"
    )

    # PPO 하이퍼파라미터
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--lr-schedule", action="store_true", help="선형 학습률 스케줄")
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)

    # 로깅/저장 설정
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-freq", type=int, default=10000)
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    parser.add_argument("--checkpoint-freq", type=int, default=50000)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

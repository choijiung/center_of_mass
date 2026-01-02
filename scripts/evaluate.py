#!/usr/bin/env python3
"""
학습된 모델 평가 스크립트 (개선 버전)

개선 사항:
1. 다중 메트릭 분리 기록 (기울기, 충격, 성공률, 제어 비용)
2. 통계적 분석 (평균, 표준편차, 백분위)
3. 다중 시드 평가 지원
4. 결과 JSON 저장
5. VecNormalize 지원
"""

import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

import numpy as np

# 프로젝트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.delivery_box_env import DeliveryBoxEnv


@dataclass
class EpisodeResult:
    """에피소드 결과"""
    reward: float
    length: int
    landing_tilt_degrees: float
    landing_velocity: float
    landing_vz: float
    impact_penalty: float
    tilt_reward: float
    success: bool
    control_cost: float
    terminated: bool  # True: 정상 종료, False: 시간 초과


@dataclass
class EvaluationStats:
    """평가 통계"""
    n_episodes: int
    success_rate: float

    # 보상 통계
    reward_mean: float
    reward_std: float
    reward_min: float
    reward_max: float

    # 기울기 통계 (도)
    tilt_mean: float
    tilt_std: float
    tilt_median: float
    tilt_p90: float  # 90 백분위수
    tilt_min: float
    tilt_max: float

    # 충격 통계
    impact_mean: float
    impact_std: float
    landing_vz_mean: float

    # 제어 비용 통계
    control_cost_mean: float
    control_cost_std: float

    # 에피소드 길이
    length_mean: float
    length_std: float


def evaluate_model(
    model_path: str,
    n_episodes: int = 100,
    vec_normalize_path: Optional[str] = None,
    xml_path: Optional[str] = None,
    deterministic: bool = True,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[List[EpisodeResult], EvaluationStats]:
    """
    학습된 모델 평가

    Args:
        model_path: 모델 파일 경로
        n_episodes: 평가할 에피소드 수
        vec_normalize_path: VecNormalize 파일 경로 (있으면)
        xml_path: MuJoCo XML 파일 경로
        deterministic: 결정적 행동 사용 여부
        seed: 랜덤 시드
        verbose: 상세 출력 여부

    Returns:
        results: 에피소드별 결과 리스트
        stats: 통계 요약
    """
    if verbose:
        print("=" * 70)
        print("택배상자 질량 중심 제어 - 모델 평가 (개선 버전)")
        print("=" * 70)
        print(f"모델: {model_path}")
        print(f"에피소드 수: {n_episodes}")
        print(f"결정적 행동: {deterministic}")
        print()

    # 환경 생성
    env = DeliveryBoxEnv(xml_path=xml_path)
    env = DummyVecEnv([lambda: env])

    # VecNormalize 로드 (있으면)
    if vec_normalize_path and Path(vec_normalize_path).exists():
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False
        if verbose:
            print(f"VecNormalize 로드됨: {vec_normalize_path}")

    # 모델 로드
    model = PPO.load(model_path, env=env)

    # 평가 결과 저장
    results: List[EpisodeResult] = []

    if verbose:
        print("\n평가 시작...")
        print("-" * 70)

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        episode_info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            step_count += 1
            episode_info = info[0]

        # 결과 기록
        result = EpisodeResult(
            reward=total_reward,
            length=step_count,
            landing_tilt_degrees=episode_info.get('landing_tilt_degrees', 90.0),
            landing_velocity=episode_info.get('landing_velocity', 0.0),
            landing_vz=episode_info.get('landing_vz', 0.0),
            impact_penalty=episode_info.get('impact_penalty', 0.0),
            tilt_reward=episode_info.get('tilt_reward', 0.0),
            success=episode_info.get('success', False),
            control_cost=episode_info.get('cumulative_control_cost', 0.0),
            terminated=episode_info.get('landed', False),
        )
        results.append(result)

        if verbose and (ep + 1) % 10 == 0:
            recent_success = sum(1 for r in results[-10:] if r.success) / 10
            recent_tilt = np.mean([r.landing_tilt_degrees for r in results[-10:]])
            print(f"  [{ep + 1:4d}/{n_episodes}] "
                  f"최근 성공률: {recent_success:.1%}, "
                  f"최근 평균 기울기: {recent_tilt:.2f}°")

    env.close()

    # 통계 계산
    rewards = [r.reward for r in results]
    tilts = [r.landing_tilt_degrees for r in results]
    impacts = [r.impact_penalty for r in results]
    landing_vzs = [r.landing_vz for r in results]
    control_costs = [r.control_cost for r in results]
    lengths = [r.length for r in results]
    successes = [r.success for r in results]

    stats = EvaluationStats(
        n_episodes=n_episodes,
        success_rate=np.mean(successes),

        reward_mean=np.mean(rewards),
        reward_std=np.std(rewards),
        reward_min=np.min(rewards),
        reward_max=np.max(rewards),

        tilt_mean=np.mean(tilts),
        tilt_std=np.std(tilts),
        tilt_median=np.median(tilts),
        tilt_p90=np.percentile(tilts, 90),
        tilt_min=np.min(tilts),
        tilt_max=np.max(tilts),

        impact_mean=np.mean(impacts),
        impact_std=np.std(impacts),
        landing_vz_mean=np.mean(landing_vzs),

        control_cost_mean=np.mean(control_costs),
        control_cost_std=np.std(control_costs),

        length_mean=np.mean(lengths),
        length_std=np.std(lengths),
    )

    if verbose:
        print()
        print("=" * 70)
        print("평가 결과 요약")
        print("=" * 70)
        print()
        print(f"{'성공률:':<25} {stats.success_rate:.1%}")
        print()
        print("[ 보상 통계 ]")
        print(f"  {'평균 ± 표준편차:':<20} {stats.reward_mean:.2f} ± {stats.reward_std:.2f}")
        print(f"  {'범위:':<20} [{stats.reward_min:.2f}, {stats.reward_max:.2f}]")
        print()
        print("[ 착지 기울기 (도) ]")
        print(f"  {'평균 ± 표준편차:':<20} {stats.tilt_mean:.2f} ± {stats.tilt_std:.2f}")
        print(f"  {'중앙값:':<20} {stats.tilt_median:.2f}")
        print(f"  {'90 백분위수:':<20} {stats.tilt_p90:.2f}")
        print(f"  {'범위:':<20} [{stats.tilt_min:.2f}, {stats.tilt_max:.2f}]")
        print()
        print("[ 착지 충격 ]")
        print(f"  {'충격 패널티:':<20} {stats.impact_mean:.2f} ± {stats.impact_std:.2f}")
        print(f"  {'평균 착지 속도 (Z):':<20} {stats.landing_vz_mean:.2f} m/s")
        print()
        print("[ 제어 비용 ]")
        print(f"  {'누적 제어 비용:':<20} {stats.control_cost_mean:.4f} ± {stats.control_cost_std:.4f}")
        print()
        print("[ 에피소드 길이 ]")
        print(f"  {'평균 스텝:':<20} {stats.length_mean:.1f} ± {stats.length_std:.1f}")
        print("=" * 70)

    return results, stats


def multi_seed_evaluation(
    model_path: str,
    seeds: List[int],
    n_episodes_per_seed: int = 50,
    **kwargs
) -> Dict[str, Any]:
    """
    다중 시드 평가 (재현성 검증)
    """
    print("=" * 70)
    print("다중 시드 평가")
    print("=" * 70)
    print(f"시드: {seeds}")
    print(f"시드당 에피소드: {n_episodes_per_seed}")
    print()

    all_stats = []

    for seed in seeds:
        print(f"\n--- 시드 {seed} ---")
        _, stats = evaluate_model(
            model_path,
            n_episodes=n_episodes_per_seed,
            seed=seed,
            verbose=False,
            **kwargs
        )
        all_stats.append(stats)
        print(f"  성공률: {stats.success_rate:.1%}, "
              f"평균 기울기: {stats.tilt_mean:.2f}°")

    # 시드 간 통계
    success_rates = [s.success_rate for s in all_stats]
    tilt_means = [s.tilt_mean for s in all_stats]

    print()
    print("=" * 70)
    print("다중 시드 통합 결과")
    print("=" * 70)
    print(f"성공률: {np.mean(success_rates):.1%} ± {np.std(success_rates):.1%}")
    print(f"평균 기울기: {np.mean(tilt_means):.2f}° ± {np.std(tilt_means):.2f}°")

    return {
        'seeds': seeds,
        'stats_per_seed': [asdict(s) for s in all_stats],
        'aggregate': {
            'success_rate_mean': np.mean(success_rates),
            'success_rate_std': np.std(success_rates),
            'tilt_mean': np.mean(tilt_means),
            'tilt_std': np.std(tilt_means),
        }
    }


def test_random_agent(n_episodes: int = 10, seed: int = 42):
    """랜덤 에이전트로 환경 테스트"""
    print("=" * 70)
    print("랜덤 에이전트 테스트")
    print("=" * 70)

    env = DeliveryBoxEnv()
    env.reset(seed=seed)

    results = []
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        tilt = info.get('landing_tilt_degrees', 90)
        success = info.get('success', False)
        results.append({'reward': total_reward, 'tilt': tilt, 'success': success})

        print(f"  에피소드 {ep + 1:3d}: 보상={total_reward:7.2f}, "
              f"기울기={tilt:6.2f}°, 성공={'O' if success else 'X'}")

    env.close()

    print()
    print("-" * 70)
    success_rate = sum(1 for r in results if r['success']) / len(results)
    avg_tilt = np.mean([r['tilt'] for r in results])
    print(f"랜덤 에이전트 성공률: {success_rate:.1%}")
    print(f"평균 착지 기울기: {avg_tilt:.2f}°")


def save_results(
    results: List[EpisodeResult],
    stats: EvaluationStats,
    save_path: str
):
    """평가 결과 저장"""
    data = {
        'stats': asdict(stats),
        'episodes': [asdict(r) for r in results],
    }

    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"결과 저장됨: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="학습된 모델 평가 (개선 버전)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model", type=str, default=None,
        help="모델 파일 경로 (.zip)"
    )
    parser.add_argument(
        "--vec-normalize", type=str, default=None,
        help="VecNormalize 파일 경로 (.pkl)"
    )
    parser.add_argument(
        "--xml-path", type=str, default=None,
        help="MuJoCo XML 파일 경로"
    )
    parser.add_argument(
        "--n-episodes", type=int, default=100,
        help="평가할 에피소드 수"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="랜덤 시드"
    )
    parser.add_argument(
        "--stochastic", action="store_true",
        help="확률적 행동 사용 (기본: 결정적)"
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="결과 저장 경로 (JSON)"
    )
    parser.add_argument(
        "--multi-seed", action="store_true",
        help="다중 시드 평가 (5개 시드)"
    )
    parser.add_argument(
        "--test-random", action="store_true",
        help="랜덤 에이전트로 환경 테스트"
    )

    args = parser.parse_args()

    if args.test_random:
        test_random_agent(n_episodes=args.n_episodes, seed=args.seed)

    elif args.model:
        if args.multi_seed:
            seeds = [42, 123, 456, 789, 1024]
            results = multi_seed_evaluation(
                args.model,
                seeds=seeds,
                n_episodes_per_seed=args.n_episodes // len(seeds),
                vec_normalize_path=args.vec_normalize,
                xml_path=args.xml_path,
                deterministic=not args.stochastic,
            )
            if args.save:
                with open(args.save, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"결과 저장됨: {args.save}")
        else:
            results, stats = evaluate_model(
                args.model,
                n_episodes=args.n_episodes,
                vec_normalize_path=args.vec_normalize,
                xml_path=args.xml_path,
                deterministic=not args.stochastic,
                seed=args.seed,
                verbose=True,
            )
            if args.save:
                save_results(results, stats, args.save)

    else:
        print("--model 또는 --test-random 옵션을 지정하세요.")
        parser.print_help()


if __name__ == "__main__":
    main()

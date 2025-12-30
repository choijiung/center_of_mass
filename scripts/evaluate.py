#!/usr/bin/env python3
"""
학습된 모델 평가 및 시각화 스크립트
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

# 프로젝트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO
from envs.delivery_box_env import DeliveryBoxEnv


def evaluate_model(model_path: str, n_episodes: int = 10, render: bool = False):
    """
    학습된 모델 평가

    Args:
        model_path: 모델 파일 경로
        n_episodes: 평가할 에피소드 수
        render: 렌더링 여부
    """
    print("=" * 60)
    print("택배상자 질량 중심 제어 - 모델 평가")
    print("=" * 60)
    print(f"모델: {model_path}")
    print(f"에피소드 수: {n_episodes}")
    print()

    # 환경 및 모델 로드
    env = DeliveryBoxEnv(render_mode="human" if render else None)
    model = PPO.load(model_path)

    # 평가 결과 저장
    results = {
        "rewards": [],
        "landing_tilts": [],
        "successes": [],
        "episode_lengths": [],
    }

    print("평가 시작...")
    print("-" * 60)

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step_count += 1

            if render:
                env.render()

        # 결과 기록
        results["rewards"].append(total_reward)
        results["episode_lengths"].append(step_count)

        if "landing_tilt_degrees" in info:
            results["landing_tilts"].append(info["landing_tilt_degrees"])
            results["successes"].append(info.get("success", False))

            print(f"에피소드 {ep + 1:3d}: "
                  f"보상={total_reward:8.2f}, "
                  f"착지 기울기={info['landing_tilt_degrees']:6.2f}°, "
                  f"성공={'O' if info['success'] else 'X'}")
        else:
            print(f"에피소드 {ep + 1:3d}: "
                  f"보상={total_reward:8.2f}, "
                  f"착지 실패 (타임아웃)")

    env.close()

    # 통계 출력
    print()
    print("-" * 60)
    print("평가 결과 요약:")
    print(f"  평균 보상: {np.mean(results['rewards']):.2f} ± {np.std(results['rewards']):.2f}")
    print(f"  평균 에피소드 길이: {np.mean(results['episode_lengths']):.1f}")

    if results["landing_tilts"]:
        print(f"  평균 착지 기울기: {np.mean(results['landing_tilts']):.2f}°")
        print(f"  성공률: {100 * np.mean(results['successes']):.1f}%")

    return results


def visualize_episode(model_path: str):
    """
    MuJoCo 뷰어로 에피소드 시각화
    """
    print("MuJoCo 뷰어로 시각화 중...")

    # 환경 및 모델 로드
    env = DeliveryBoxEnv()
    model = PPO.load(model_path)

    obs, info = env.reset()

    # MuJoCo 뷰어 실행
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.distance = 5.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 45

        done = False
        while viewer.is_running() and not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            viewer.sync()

        # 착지 후 잠시 유지
        if viewer.is_running():
            import time
            for _ in range(100):
                mujoco.mj_step(env.model, env.data)
                viewer.sync()
                time.sleep(0.01)

    env.close()


def test_random_agent(n_episodes: int = 5):
    """
    랜덤 에이전트로 환경 테스트
    """
    print("=" * 60)
    print("랜덤 에이전트 테스트")
    print("=" * 60)

    env = DeliveryBoxEnv()

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        print(f"에피소드 {ep + 1}: 보상={total_reward:.2f}")
        if "landing_tilt_degrees" in info:
            print(f"  -> 착지 기울기: {info['landing_tilt_degrees']:.2f}°")

    env.close()


def main():
    parser = argparse.ArgumentParser(
        description="학습된 모델 평가 및 시각화"
    )

    parser.add_argument(
        "--model", type=str, default=None,
        help="모델 파일 경로 (.zip)"
    )
    parser.add_argument(
        "--n-episodes", type=int, default=10,
        help="평가할 에피소드 수 (기본값: 10)"
    )
    parser.add_argument(
        "--render", action="store_true",
        help="렌더링 활성화"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="MuJoCo 뷰어로 시각화"
    )
    parser.add_argument(
        "--test-random", action="store_true",
        help="랜덤 에이전트로 환경 테스트"
    )

    args = parser.parse_args()

    if args.test_random:
        test_random_agent(args.n_episodes)
    elif args.model:
        if args.visualize:
            visualize_episode(args.model)
        else:
            evaluate_model(args.model, args.n_episodes, args.render)
    else:
        print("--model 또는 --test-random 옵션을 지정하세요.")
        parser.print_help()


if __name__ == "__main__":
    main()

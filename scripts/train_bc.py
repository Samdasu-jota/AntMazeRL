"""
train_bc.py — BC 학습 실행 + 학습곡선 저장 + 환경 평가 (2.2).

실행: 프로젝트 루트에서  python -m scripts.train_bc
설정: configs/bc_config.yaml
"""

import os
os.environ.setdefault("MUJOCO_GL", "glfw")

import yaml
import matplotlib
matplotlib.use("Agg")                       # 헤드리스 환경에서 png 저장
import matplotlib.pyplot as plt
import gymnasium as gym
import src.envs                              # AntMaze-v0 등록
from src.imitation.behavior_cloning import train_bc, evaluate_policy


def main():
    # 설정 로드
    with open("configs/bc_config.yaml") as f:
        config = yaml.safe_load(f)

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # 1) 학습
    print("BC 학습 시작...\n")
    train_losses, val_losses = train_bc(config)

    # 2) 학습 곡선 저장
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="train loss")
    plt.plot(val_losses, label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.title("Behavior Cloning training curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(config["curve_out"], bbox_inches="tight", dpi=120)
    plt.close()
    print(f"\n✅ 학습곡선 저장: {config['curve_out']}")

    # 3) 환경에서 평가 (20 에피소드)
    print("\n학습된 정책 평가 (20 에피소드)...")
    env = gym.make("AntMaze-v0")
    avg_reward, success_rate = evaluate_policy(
        env, config["model_out"],
        n_episodes=20, hidden_dim=config["hidden_dim"],
    )
    env.close()

    print("\n── BC 정책 평가 ──────────────")
    print(f"  평균 보상 : {avg_reward:.2f}")
    print(f"  성공률    : {success_rate * 100:.1f}%")
    print(f"\n✅ 모델 저장됨: {config['model_out']}")


if __name__ == "__main__":
    main()

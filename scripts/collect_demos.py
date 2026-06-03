"""
collect_demos.py — 전문가 데모 200 에피소드 수집 → data/expert_demos.npz 저장 (2.1).

먼저 scripts/preview_expert.py 로 걸음새(특히 C단계 waypoint)가 목표에 도달하는지
확인한 뒤 실행하는 게 좋다. 여기서 모은 (obs, action) 쌍이 2.2 behavior cloning의
학습 데이터가 된다.

실행 (프로젝트 루트에서):  python -m scripts.collect_demos
"""

import os
os.environ.setdefault("MUJOCO_GL", "glfw")

import numpy as np
import gymnasium as gym
import src.envs                              # ← AntMaze-v0 등록 (필수)
from src.experts.scripted_expert import collect_demonstrations


def main():
    env = gym.make("AntMaze-v0")             # 렌더 불필요 (데이터만)

    print("전문가 데모 수집 시작 (200 에피소드, mode='waypoint')...\n")
    data = collect_demonstrations(env, n_episodes=200, mode="waypoint")
    env.close()

    os.makedirs("data", exist_ok=True)
    np.savez_compressed(
        "data/expert_demos.npz",
        observations=data["observations"],
        actions=data["actions"],
    )

    s = data["stats"]
    print("\n── 수집 완료 ──────────────────────")
    print(f"  성공률(목표 도달)     : {s['success_rate'] * 100:.1f}%")
    print(f"  전복률(넘어져 못 일어남): {s['topple_rate'] * 100:.1f}%")
    print(f"  평균 스텝             : {s['avg_steps']:.1f}")
    print(f"  평균 보상             : {s['avg_reward']:.2f}")
    print(f"  저장된 전이 수(서있는 것만): {s['total_transitions']}")
    print(f"\n✅ 저장: data/expert_demos.npz")
    print(f"   observations: {data['observations'].shape}")
    print(f"   actions     : {data['actions'].shape}")
    print("\n   ※ 넘어진 상태의 버둥거림은 제외하고 '서서 움직이는' 전이만 저장됩니다.")

    if s["total_transitions"] == 0:
        print("\n⚠️  저장된 전이 0개 — 매번 즉시 넘어졌습니다.")
        print("    scripts/preview_expert.py 로 A→B→C 단계별로 보며")
        print("    src/experts/scripted_expert.py 상단 상수를 튜닝하세요.")


if __name__ == "__main__":
    main()

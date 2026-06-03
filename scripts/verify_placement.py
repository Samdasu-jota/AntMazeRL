"""
가설 검증: 미로 막힘이 '기둥 배치/goal 위치' 문제인가, '달리며 꺾기' 한계인가?

frozen turn30 워커를 여러 조건에서 eval-only로 비교 (학습 없음):
  E1  기둥 격리   — 동일 waypoints로 미로(기둥) vs 빈 평지(기둥 없음)
  E2  goal 반경   — 미로, WaypointFollower(goal_radius) ∈ {1.0, 1.5, 2.0}
  E3  goal 위치   — 미로, 최종 goal을 벽 끝에서 뗌 ((0,6.8) 위, (-1.5,6) 옆)

각 조건: 성공률 + wp_idx별 정체 분포 + 종료원인(reached/fell/timeout).
사용: venv/bin/python -m scripts.verify_placement
"""
import os
os.environ.setdefault("MUJOCO_GL", "glfw")

from collections import Counter
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

import src.envs  # noqa: F401
from src.envs.waypoint_follower import WaypointFollower
from src.envs.wrappers import FixedNormalizeObs
from src.training.train_ppo import load_norm_stats

MODEL = "models/checkpoints_stage1_cmdfollow_turn30/ppo_final.zip"
DEFAULT_WPS = [(2.5, 0.0), (2.5, 6.3), (0.0, 6.0)]   # ant_maze_env.WAYPOINTS와 동일
OM, OSD = load_norm_stats("data/obs_norm_stats.npz")


def evaluate(model, env_id, waypoints, goal_radius=1.0, n=60, seed0=20000):
    """조건 1개 평가 → (success_rate, wp_idx 분포 Counter, 종료원인 Counter, mean_wp)."""
    base = gym.make(env_id)
    env = FixedNormalizeObs(
        WaypointFollower(base, waypoints=[np.asarray(w, float) for w in waypoints],
                         goal_radius=goal_radius), OM, OSD)
    succ, wpc, cause, wps = 0, Counter(), Counter(), []
    for ep in range(n):
        obs, info = env.reset(seed=seed0 + ep)
        done, mwp, term = False, 0, False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(a)
            mwp = max(mwp, info.get("wp_idx", 0))
            done = term or trunc
        wpc[mwp] += 1
        wps.append(mwp)
        if info.get("reached_goal"):
            succ += 1
            cause["reached"] += 1
        elif term:
            cause["fell"] += 1
        else:
            cause["timeout"] += 1
    env.close()
    return succ / n, wpc, cause, float(np.mean(wps))


def show(label, res):
    sr, wpc, cause, mwp = res
    dist = " ".join(f"wp{k}:{wpc[k]}" for k in sorted(wpc))
    print(f"  {label:38s} 성공 {sr*100:5.1f}% | mean_wp {mwp:.2f} | {dist} | {dict(cause)}")


def main():
    model = PPO.load(MODEL, device="cpu")
    N = 60
    print(f"\n검증 대상: {MODEL}  (frozen, {N} eps/조건)\n")

    print("E1 — 기둥 격리 (동일 waypoints, 미로 vs 빈 평지) ⭐ 결정적")
    show("(a) 미로(기둥) r=1.0", evaluate(model, "AntMazeWaypoint-v0", DEFAULT_WPS, 1.0, N))
    show("(b) 빈 평지(기둥X) r=1.0", evaluate(model, "AntMazeOpen-v0", DEFAULT_WPS, 1.0, N))

    print("\nE2 — goal 반경 sweep (미로)")
    for r in (1.0, 1.5, 2.0):
        show(f"미로 goal_radius={r}", evaluate(model, "AntMazeWaypoint-v0", DEFAULT_WPS, r, N))

    print("\nE3 — goal 위치 이동 (미로, 최종 goal을 벽 끝에서 뗌)")
    wps_up = [(2.5, 0.0), (2.5, 6.8), (0.0, 6.8)]      # 벽(y=6) 위쪽으로
    wps_side = [(2.5, 0.0), (2.5, 6.0), (-1.5, 6.0)]   # 왼쪽 측면(inflated 벽 밖)
    show("미로 goal (0,6.8) 벽 위쪽", evaluate(model, "AntMazeWaypoint-v0", wps_up, 1.0, N))
    show("미로 goal (-1.5,6) 측면", evaluate(model, "AntMazeWaypoint-v0", wps_side, 1.0, N))


if __name__ == "__main__":
    main()

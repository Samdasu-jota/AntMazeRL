"""
Stage 1 — 전방향(omnidirectional) 워커 평가: 8방위(octant) × 거리별 성공률.

Stage 0 워커는 +y만 학습한 '방향맹'이었다. fine-tune 후 '어느 방향이든 조향'하는지를
방위별로 쪼개 확인한다(+y 편향 잔존 여부). 목표는 경계벽 안 reachable 거리만 테스트.

사용:
  python -m scripts.eval_omnidir --checkpoint models/checkpoints_stage1_omnidir/ppo_final.zip \
      --reps 6
"""
import os
os.environ.setdefault("MUJOCO_GL", "glfw")

import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

import src.envs  # noqa: F401  (AntMazeOpen-v0 등록)
from src.envs.wrappers import FixedNormalizeObs
from src.training.train_ppo import load_norm_stats

BOX = (-3.4, 3.4, -1.4, 7.4)          # 경계벽 안쪽 reachable box
OCTANTS = ["E(+x)", "NE", "N(+y)", "NW", "W(-x)", "SW", "S(-y)", "SE"]


def max_reach(ang, box=BOX):
    c, s = np.cos(ang), np.sin(ang)
    xmin, xmax, ymin, ymax = box
    ts = []
    if c > 1e-6:
        ts.append(xmax / c)
    elif c < -1e-6:
        ts.append(xmin / c)
    if s > 1e-6:
        ts.append(ymax / s)
    elif s < -1e-6:
        ts.append(ymin / s)
    return min(ts) if ts else 6.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--reps", type=int, default=6, help="방위·거리당 에피소드 수")
    ap.add_argument("--norm-stats", default="data/obs_norm_stats.npz")
    args = ap.parse_args()

    obs_mean, obs_std = load_norm_stats(args.norm_stats)
    model = PPO.load(args.checkpoint, device="cpu")
    env = gym.make("AntMazeOpen-v0")
    env = FixedNormalizeObs(env, obs_mean, obs_std)
    u = env.unwrapped

    seed = 30000
    tot_succ = tot = 0
    print(f"\n방위별 성공률 (reps={args.reps}/거리, 거리는 reachable 안에서 3점):")
    worst = 1.0
    for oi, ang in enumerate(np.arange(8) * (np.pi / 4)):
        M = max_reach(ang)
        # reachable 안에서 가까움/중간/멀음 3거리 (M이 작으면 좁게)
        hi = min(5.0, M - 0.3)
        lo = min(1.5, 0.6 * hi) if hi > 0 else 0.5
        dists = np.linspace(lo, hi, 3) if hi > lo else [max(0.5, 0.7 * M)]
        o_succ = o_tot = 0
        for d in dists:
            goal = np.array([d * np.cos(ang), d * np.sin(ang)])
            for _ in range(args.reps):
                u.goal_pos = goal.copy()
                u._prev_goal_dist = None
                obs, info = env.reset(seed=seed)
                seed += 1
                done = False
                while not done:
                    a, _ = model.predict(obs, deterministic=True)
                    obs, _, term, trunc, info = env.step(a)
                    done = term or trunc
                o_tot += 1
                if info.get("reached_goal", False):
                    o_succ += 1
        sr = o_succ / max(o_tot, 1)
        worst = min(worst, sr)
        tot_succ += o_succ
        tot += o_tot
        print(f"  {OCTANTS[oi]:>7}  성공 {o_succ:>2}/{o_tot:<2}  ({sr*100:5.1f}%)  "
              f"[거리 {np.round(dists,1).tolist()}]")
    env.close()
    overall = tot_succ / max(tot, 1)
    gate = (overall >= 0.80 and worst >= 0.60)
    print(f"\n전체 성공률 {overall*100:.1f}% | 최악 방위 {worst*100:.1f}% "
          f"→ Phase-2 게이트(≥80% & 방위별 ≥60%) {'PASS ✅' if gate else 'FAIL ❌'}")


if __name__ == "__main__":
    main()

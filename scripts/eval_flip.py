"""
Phase 0 진단 — up_z(자세) 기반 '뒤집힘' 측정. 학습/환경 코드 무수정(읽기 전용).

기존 종료조건은 z<0.2(키)만 봐서 '뒤집힌 채 기어가기'(z>0.2 유지)를 못 잡는다.
미로 진단의 'fell 0/40'은 z만 본 것이라 뒤집힘에 눈먼 지표다.
이 스크립트는 체크포인트를 불러와 매 스텝 up_z = 1 - 2*(qx²+qy²)를 직접 계산해
정말 뒤집히는지/얼마나 자주인지를 정량화한다. env 편집 없이 unwrapped qpos만 읽으므로
검증된 87% 워커의 재현성을 전혀 건드리지 않는다.

up_z 해석: +1 똑바로 섬 · 0 옆으로 누움(90° 기울기) · -1 완전히 뒤집힘(배가 위로).

사용:
  python -m scripts.eval_flip --config configs/ppo_stage0_scratch_6m.yaml \
      --checkpoint models/checkpoints_stage0_scratch_6m/ppo_final.zip --n 100
  python -m scripts.eval_flip --config configs/ppo_stage1_maze_short.yaml \
      --checkpoint models/checkpoints_stage1_maze_short/ppo_final.zip --n 100
"""
import os
os.environ.setdefault("MUJOCO_GL", "glfw")

import argparse
from collections import Counter

import numpy as np
import yaml
import gymnasium as gym
from stable_baselines3 import PPO

import src.envs  # noqa: F401  (AntMaze* 등록)
from src.envs.wrappers import FixedNormalizeObs
from src.training.train_ppo import load_norm_stats, apply_env_wrapper


def up_z_from_qpos(qpos):
    """몸통 up축의 world z성분. qpos[3:7]=쿼터니언(qw,qx,qy,qz) → up_z = 1 - 2*(qx²+qy²).
    env의 종료/보상 코드와 동일한 공식(향후 _flipped와 정확히 일치)."""
    qx, qy = qpos[4], qpos[5]
    return 1.0 - 2.0 * (qx * qx + qy * qy)


def build_env(cfg, obs_mean, obs_std, seed=0):
    """학습(make_env)과 동일하게 env 재구성: gym.make → 래퍼 → FixedNormalizeObs."""
    env = gym.make(cfg.get("env_id", "AntMaze-v0"), **(cfg.get("env_kwargs") or {}))
    env = apply_env_wrapper(env, cfg.get("env_wrapper"),
                            cfg.get("env_wrapper_kwargs") or {}, seed=seed)
    env = FixedNormalizeObs(env, obs_mean, obs_std)
    return env


def run(model, env, n, flip_thresh, deterministic=True, seed0=20000):
    u = env.unwrapped                       # 기본 AntMazeEnv (data.qpos 보유)
    rows = []
    for ep in range(n):
        obs, info = env.reset(seed=seed0 + ep)
        done, steps = False, 0
        min_uz, n_inverted = 1.0, 0
        last_uz, last_z = 1.0, None
        term = trunc = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, term, trunc, info = env.step(action)
            qpos = u.data.qpos
            uz = up_z_from_qpos(qpos)
            min_uz = min(min_uz, uz)
            if uz < 0.0:                     # 배가 옆/위로 (옆으로 누움 이상)
                n_inverted += 1
            last_uz, last_z = uz, float(qpos[2])
            steps += 1
            done = term or trunc
        reached = bool(info.get("reached_goal", False))
        if reached:
            cause = "reached"
        elif trunc and not term:
            cause = "timeout"
        elif last_z is not None and last_z < 0.2:
            cause = "fell_z"                 # 기존 z<0.2 종료
        else:
            cause = "term_other"
        rows.append(dict(
            steps=steps, reached=reached, min_uz=min_uz,
            frac_inv=n_inverted / max(steps, 1), last_uz=last_uz, cause=cause,
            flipped=(min_uz < flip_thresh),
            ended_inverted=(term and not reached and last_uz < flip_thresh),
        ))
    return rows


def summarize(name, rows, flip_thresh):
    n = len(rows)
    succ = float(np.mean([r["reached"] for r in rows]))
    mean_len = float(np.mean([r["steps"] for r in rows]))
    flip_rate = float(np.mean([r["flipped"] for r in rows]))
    flip_term_rate = float(np.mean([r["ended_inverted"] for r in rows]))
    mean_min_uz = float(np.mean([r["min_uz"] for r in rows]))
    mean_frac_inv = float(np.mean([r["frac_inv"] for r in rows]))
    succ_min = [r["min_uz"] for r in rows if r["reached"]]
    worst_walking = float(min(succ_min)) if succ_min else float("nan")
    causes = Counter(r["cause"] for r in rows)

    print(f"\n===== {name}  (n={n}, flip_thresh={flip_thresh}) =====")
    print(f"  성공률            {succ * 100:5.1f}%")
    print(f"  평균 ep_len       {mean_len:6.0f}")
    print(f"  flip_rate         {flip_rate * 100:5.1f}%   (에피소드 중 min up_z < {flip_thresh})")
    print(f"  flip_term_rate    {flip_term_rate * 100:5.1f}%   (뒤집힌 채 종료, 도달X)")
    print(f"  mean min up_z     {mean_min_uz:6.3f}")
    print(f"  평균 뒤집힘 비율  {mean_frac_inv * 100:5.1f}%   (up_z<0 스텝 비율)")
    print(f"  성공ep 최저 up_z  {worst_walking:6.3f}   (정상보행 최대 기울기 → UP_THRESH는 이보다 낮게)")
    print(f"  종료원인          {dict(causes)}")

    bins = [-1.01, -0.5, 0.0, 0.3, 0.5, 0.7, 0.9, 1.01]
    labels = ["[-1.0,-0.5)", "[-0.5, 0.0)", "[ 0.0, 0.3)", "[ 0.3, 0.5)",
              "[ 0.5, 0.7)", "[ 0.7, 0.9)", "[ 0.9, 1.0]"]
    hist, _ = np.histogram([r["min_uz"] for r in rows], bins=bins)
    print("  per-ep min up_z 분포:")
    for lab, h in zip(labels, hist):
        bar = "█" * int(round(40 * h / max(n, 1)))
        print(f"    {lab} {h:3d} {bar}")

    return dict(name=name, n=n, success=succ, mean_len=mean_len,
                flip_rate=flip_rate, flip_term_rate=flip_term_rate,
                mean_min_uz=mean_min_uz, mean_frac_inv=mean_frac_inv,
                worst_walking=worst_walking, causes=dict(causes))


def main():
    ap = argparse.ArgumentParser(description="up_z 뒤집힘 진단 (eval-only, env 무수정)")
    ap.add_argument("--config", required=True, help="학습 config YAML (env 재구성용)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--flip-thresh", type=float, default=0.0,
                    help="이 값 밑으로 내려가면 '뒤집힘'으로 셈 (기본 0.0 = 옆으로 누움 이상)")
    ap.add_argument("--stochastic", action="store_true",
                    help="결정적 대신 확률적 롤아웃(학습 중 행동에 더 가까움)")
    ap.add_argument("--norm-stats", default=None, help="기본=config의 norm_stats_path")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    norm_path = args.norm_stats or cfg.get("norm_stats_path", "data/obs_norm_stats.npz")
    obs_mean, obs_std = load_norm_stats(norm_path)
    model = PPO.load(args.checkpoint, device="cpu")

    env = build_env(cfg, obs_mean, obs_std)
    rows = run(model, env, args.n, args.flip_thresh, deterministic=not args.stochastic)
    env.close()
    tag = "stochastic" if args.stochastic else "deterministic"
    summarize(f"{os.path.basename(args.checkpoint)} @ {cfg.get('env_id')} [{tag}]",
              rows, args.flip_thresh)


if __name__ == "__main__":
    main()

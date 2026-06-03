"""
Phase 3.3 — 정식 평가·비교 (random vs BC vs RL + 팩트별 임팩트 ablation).

README 파이프라인 5단계("평가 & 문서화 — random vs BC vs RL 성능 비교")의 구현.
좁은 3자 비교가 아니라, 일지의 핵심 '개입(팩트)'들이 성능을 얼마나 올렸는지를
**한 하니스로** 측정한다. 각 팩트 Δ는 env를 고정하고 한 변수만 바꿔 apples-to-apples.

핵심 배선(중요):
  - PPO : FixedNormalizeObs(고정 통계)로 정규화한 obs → model.predict(deterministic).
  - BC  : BCPolicy는 forward 안에서 자체 정규화 → **raw obs**(정규화 래퍼 없이) 먹여야 함.
  - random: env.action_space.sample().
모두 같은 seed(seed0=20000), 100ep, 결정적. 58/88/73/39/87 재현이 곧 하니스 검증.

사용:
  python -m scripts.evaluate_comparison [--n 100] [--out outputs/evaluation_results.json]
"""
import os
os.environ.setdefault("MUJOCO_GL", "glfw")

import argparse
import datetime
import json

import numpy as np
import yaml
import torch
import gymnasium as gym
from stable_baselines3 import PPO

import src.envs  # noqa: F401  (AntMaze* 등록)
from src.envs.wrappers import FixedNormalizeObs
from src.training.train_ppo import load_norm_stats, apply_env_wrapper
from src.imitation.behavior_cloning import BCPolicy, get_device
from scripts.eval_flip import up_z_from_qpos


# ── 평가 대상 (label, kind, checkpoint, config) ──────────────────────────────
#   group/env_label = 어느 무대(맵)에서 잰 숫자인가 (도표·정직성용).
#   ⚠️ regime: 자세종료(up_thresh<up_z)는 flip-fix의 *일부*다. env 현재 기본값 up_thresh=0.0이라
#      그대로 두면 '전복수정 前' 정책(뒤집혀 기어가던)을 부당하게 즉시 종료시켜 일지값(58/81/87)을
#      못 재현한다. 그래서 전복수정 前 정책은 env_override={'up_thresh':-1.1}로 *원래 측정 regime*
#      (자세종료 OFF, z<0.2만 종료)을 복원한다. 전복수정 後 정책은 up_thresh=0.0(자세종료 ON) 유지.
PRE = {"up_thresh": -1.1}   # 전복수정 前 regime: 자세종료 OFF (up_z는 항상 ≥-1.0 → 절대 안 걸림)
ENTRIES = [
    dict(key="random_short", label="random", kind="random",
         checkpoint=None, config="configs/ppo_p3_maze_upright.yaml",
         group="maze_short", env_label="축소미로 honest r1.0"),
    dict(key="bc_short", label="BC (모방학습)", kind="bc",
         checkpoint="models/bc_policy.pt", config="configs/ppo_p3_maze_upright.yaml",
         group="maze_short", env_label="축소미로 honest r1.0"),
    dict(key="rl_preflip_short", label="RL 전복前", kind="ppo",
         checkpoint="models/checkpoints_stage1_maze_short/ppo_final.zip",
         config="configs/ppo_stage1_maze_short.yaml", env_override=PRE,
         group="maze_short", env_label="축소미로 honest r1.0"),
    dict(key="rl_postflip_short", label="RL 전복後(직립)", kind="ppo",
         checkpoint="models/checkpoints_p3_maze_upright/ppo_final.zip",
         config="configs/ppo_p3_maze_upright.yaml",
         group="maze_short", env_label="축소미로 honest r1.0"),
    dict(key="rl_upright_full_zeroshot", label="직립워커 → 풀맵 (zero-shot)", kind="ppo",
         checkpoint="models/checkpoints_p3_maze_upright/ppo_final.zip",
         config="configs/eval_p3_full_honest.yaml",
         group="maze_full", env_label="풀맵 honest r1.0"),
    dict(key="rl_full_finetune", label="풀맵 직접학습 (fine-tune)", kind="ppo",
         checkpoint="models/checkpoints_p3_maze_fullpillar/ppo_final.zip",
         config="configs/ppo_p3_maze_fullpillar.yaml",
         group="maze_full", env_label="풀맵 honest r1.0"),
    dict(key="scratch_plane3", label="scratch (BC 없이)", kind="ppo",
         checkpoint="models/checkpoints_stage0_scratch/ppo_final.zip",
         config="configs/ppo_stage0_scratch.yaml", env_override=PRE,
         group="plane3", env_label="평지 3m (open)"),
    dict(key="warm_plane3", label="BC warm-start", kind="ppo",
         checkpoint="models/checkpoints_stage0_warm/ppo_final.zip",
         config="configs/ppo_stage0_warm.yaml", env_override=PRE,
         group="plane3", env_label="평지 3m (open)"),
    dict(key="scratch_plane6", label="scratch 6m (직립 seed)", kind="ppo",
         checkpoint="models/checkpoints_stage0_scratch_6m/ppo_final.zip",
         config="configs/ppo_stage0_scratch_6m.yaml", env_override=PRE,
         group="plane6", env_label="평지 6m (open)"),
]


def build_env_ppo(cfg):
    """PPO 평가용: gym.make → 래퍼 → FixedNormalizeObs (eval_flip.build_env과 동일)."""
    norm_path = cfg.get("norm_stats_path", "data/obs_norm_stats.npz")
    obs_mean, obs_std = load_norm_stats(norm_path)
    env = gym.make(cfg.get("env_id", "AntMaze-v0"), **(cfg.get("env_kwargs") or {}))
    env = apply_env_wrapper(env, cfg.get("env_wrapper"),
                            cfg.get("env_wrapper_kwargs") or {}, seed=0)
    return FixedNormalizeObs(env, obs_mean, obs_std)


def build_env_raw(cfg):
    """BC/random 평가용: 정규화 래퍼 '없이' raw obs (BC는 내부에서 자체 정규화)."""
    env = gym.make(cfg.get("env_id", "AntMaze-v0"), **(cfg.get("env_kwargs") or {}))
    return apply_env_wrapper(env, cfg.get("env_wrapper"),
                             cfg.get("env_wrapper_kwargs") or {}, seed=0)


def make_policy_fn(entry, env):
    """kind에 맞는 행동 함수 반환."""
    if entry["kind"] == "random":
        env.action_space.seed(12345)
        return lambda obs: env.action_space.sample()
    if entry["kind"] == "ppo":
        model = PPO.load(entry["checkpoint"], device="cpu")
        return lambda obs: model.predict(obs, deterministic=True)[0]
    if entry["kind"] == "bc":
        device = get_device()
        bc = BCPolicy()
        bc.load_state_dict(torch.load(entry["checkpoint"], map_location=device))
        bc.to(device).eval()

        def bc_fn(obs):
            with torch.no_grad():
                t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                return bc(t).cpu().numpy()[0]
        return bc_fn
    raise ValueError(entry["kind"])


def run_entry(entry, n, seed0=20000):
    cfg = yaml.safe_load(open(entry["config"]))
    # 전복수정 前 정책은 원래 측정 regime(자세종료 OFF) 복원: up_thresh 등 override 병합.
    if entry.get("env_override"):
        cfg["env_kwargs"] = {**(cfg.get("env_kwargs") or {}), **entry["env_override"]}
    up_thresh_eff = (cfg.get("env_kwargs") or {}).get("up_thresh", 0.0)
    env = build_env_ppo(cfg) if entry["kind"] == "ppo" else build_env_raw(cfg)
    u = env.unwrapped
    policy_fn = make_policy_fn(entry, env)

    rows = []
    for ep in range(n):
        obs, info = env.reset(seed=seed0 + ep)
        done, steps, min_uz, n_inv, ret = False, 0, 1.0, 0, 0.0
        while not done:
            obs, r, term, trunc, info = env.step(policy_fn(obs))
            uz = up_z_from_qpos(u.data.qpos)
            min_uz = min(min_uz, uz)
            n_inv += int(uz < 0.0)
            ret += float(r)
            steps += 1
            done = term or trunc
        rows.append(dict(reached=bool(info.get("reached_goal", False)),
                         steps=steps, min_uz=min_uz,
                         frac_inv=n_inv / max(steps, 1), ret=ret))
    env.close()

    success = float(np.mean([x["reached"] for x in rows]))
    res = dict(
        label=entry["label"], kind=entry["kind"], group=entry["group"],
        env_label=entry["env_label"], checkpoint=entry["checkpoint"],
        config=entry["config"], env_id=cfg.get("env_id"),
        env_kwargs=cfg.get("env_kwargs") or {},
        posture_term=bool(up_thresh_eff > -1.0),   # True=자세종료 ON(전복수정 後 regime)
        regime=("flip-fix ON (자세종료)" if up_thresh_eff > -1.0 else "pre-flip (자세종료 OFF)"),
        success_rate=round(success * 100, 1),
        mean_ep_len=round(float(np.mean([x["steps"] for x in rows])), 1),
        flip_rate=round(float(np.mean([x["min_uz"] < 0 for x in rows])) * 100, 1),
        mean_min_up_z=round(float(np.mean([x["min_uz"] for x in rows])), 3),
        frac_inverted=round(float(np.mean([x["frac_inv"] for x in rows])) * 100, 1),
        mean_reward=round(float(np.mean([x["ret"] for x in rows])), 1),
        n_episodes=n,
    )
    return res


def build_factor_deltas(R):
    """측정된 엔트리(R: key→res)에서 '한 변수만 바꾼' 통제 Δ를 계산.
    F3(기하)만은 통제 짝의 옛 체크포인트가 다른 학습이라 일지값(풀 honest ~21%)을 인용."""
    def sr(k):
        return R[k]["success_rate"]
    deltas = [
        dict(key="F1_learning", label="학습 자체 (random→RL)", env="축소미로",
             before=("random", sr("random_short")),
             after=("RL 전복後", sr("rl_postflip_short")),
             delta=round(sr("rl_postflip_short") - sr("random_short"), 1),
             source="measured"),
        dict(key="F2_bc_vs_scratch", label="모방학습(BC) vs scratch", env="평지 3m",
             before=("BC warm-start", sr("warm_plane3")),
             after=("scratch", sr("scratch_plane3")),
             delta=round(sr("scratch_plane3") - sr("warm_plane3"), 1),
             source="measured", note="양수=scratch가 더 나음 (BC가 오히려 해)"),
        dict(key="F3_geometry", label="기하: 풀맵→축소기둥+A* (전복수정 前)", env="미로 honest r1.0",
             before=("풀맵 honest", 21.0),
             after=("축소+A*", sr("rl_preflip_short")),
             delta=round(sr("rl_preflip_short") - 21.0, 1),
             source="log(풀 honest 21%)+measured(축소 58%)"),
        dict(key="F4_flipfix", label="★ 전복수정 (flip-fix)", env="축소미로 (동일!)",
             before=("RL 전복前 58%·flip99%", sr("rl_preflip_short")),
             after=("RL 전복後 88%·flip11%", sr("rl_postflip_short")),
             delta=round(sr("rl_postflip_short") - sr("rl_preflip_short"), 1),
             source="measured", note="env 완전 동일 — 자세종료+직립보너스만 추가"),
        dict(key="F5_map_difficulty", label="맵 난이도: 축소→풀맵 (zero-shot)", env="직립워커 동일정책",
             before=("축소", sr("rl_postflip_short")),
             after=("풀맵 zero-shot", sr("rl_upright_full_zeroshot")),
             delta=round(sr("rl_upright_full_zeroshot") - sr("rl_postflip_short"), 1),
             source="measured", note="음수=맵이 어려워짐(퇴행 아님)"),
        dict(key="F6_full_finetune", label="풀맵 직접학습 (일반화)", env="풀맵 honest r1.0",
             before=("zero-shot", sr("rl_upright_full_zeroshot")),
             after=("fine-tune", sr("rl_full_finetune")),
             delta=round(sr("rl_full_finetune") - sr("rl_upright_full_zeroshot"), 1),
             source="measured"),
    ]
    return deltas


def main():
    ap = argparse.ArgumentParser(description="Phase 3.3 정식 평가·비교 (random/BC/RL + 팩트 ablation)")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="outputs/evaluation_results.json")
    args = ap.parse_args()

    R = {}
    for e in ENTRIES:
        print(f"\n▶ {e['key']}  ({e['kind']} @ {e['env_label']}) …", flush=True)
        res = run_entry(e, args.n)
        R[e["key"]] = res
        print(f"   성공 {res['success_rate']:.1f}% | ep_len {res['mean_ep_len']:.0f} | "
              f"flip {res['flip_rate']:.1f}% | min_up_z {res['mean_min_up_z']:.3f} | "
              f"reward {res['mean_reward']:.0f}")

    out = dict(
        meta=dict(n_episodes=args.n, seed0=20000, deterministic=True,
                  date=str(datetime.date.today()),
                  note="PPO=FixedNormalizeObs, BC=raw obs(내부정규화), random=action_space.sample. "
                       "같은 env서 한 변수만 바꾼 Δ만 통제 비교."),
        entries=R,
        factor_deltas=build_factor_deltas(R),
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 저장: {args.out}")

    print("\n=== 팩트별 임팩트 (Δ 성공률 pp) ===")
    for d in out["factor_deltas"]:
        print(f"  {d['delta']:+6.1f}pp  {d['label']:32s} [{d['env']}]")


if __name__ == "__main__":
    main()

"""
Phase 4.1 — 학습된 정책으로 rollout 수집 → 월드모델 데이터셋(.npz).

축소맵 직립 워커(models/checkpoints_p3_maze_upright/ppo_final.zip, 미로 ~88%)를
AntMazeWaypoint-v0(+waypoint_follow)에서 굴려 (state, action, next_state, reward, done)
전이를 저장한다. Phase 4.2 Transformer World Model의 학습 데이터.

핵심 설계 (plan: vs-code-wobbly-matsumoto.md):
  - RAW obs를 저장한다. 정규화(FixedNormalizeObs)는 obs를 ±10으로 '클립'하므로 역변환이
    손실적 → 정규화 래퍼 '없이'(evaluate_comparison.build_env_raw 패턴) env를 만들어
    env.step/reset이 raw 109차원 obs를 그대로 주게 하고, 정책 입력용으로만 사본을
    정규화한다:  obs_norm = clip((raw-mean)/(std+1e-6), -10, 10).
  - WaypointFollower는 매 스텝 goal_pos를 현재 서브목표로 바꾸므로 obs[107:109](목표
    상대벡터)가 웨이포인트 전환 시 '점프'한다. 활성 _wp_idx는 obs에 없는 은닉변수 →
    그 전이는 (obs,a)→obs'가 Markov가 아니다. wp_idx/subgoal_xy/wp_switch를 함께 저장해
    4.2가 drop/mask/재계산을 고르게 한다.
  - 행동 모드 혼합: 앞 n_det 에피소드는 결정적(로그 88% 재현 가능), 나머지는 확률적
    (커버리지). action_mode로 전이마다 태그.

실행:
  python -m scripts.collect_rollouts --episodes 1000 --det-frac 0.5
  python -m scripts.collect_rollouts --smoke        # 2 에피소드 + 무결성 검증, 로그/저장 분리
"""
import os
os.environ.setdefault("MUJOCO_GL", "glfw")

import argparse
import json
import subprocess
from datetime import datetime, timezone

import numpy as np
import yaml
import gymnasium as gym
from stable_baselines3 import PPO

import src.envs  # noqa: F401  (AntMaze* 등록)
from src.training.train_ppo import load_norm_stats, apply_env_wrapper

CLIP = 10.0  # FixedNormalizeObs 기본 clip — 정책이 본 입력을 정확히 재현하려면 동일해야 함


def build_env_raw(cfg, seed=0):
    """정규화 래퍼 '없이' env 재구성 → env.step/reset이 RAW obs를 반환.
    evaluate_comparison.build_env_raw와 동일(gym.make → apply_env_wrapper)."""
    env = gym.make(cfg.get("env_id", "AntMaze-v0"), **(cfg.get("env_kwargs") or {}))
    return apply_env_wrapper(env, cfg.get("env_wrapper"),
                             cfg.get("env_wrapper_kwargs") or {}, seed=seed)


def normalize(raw, mean, std):
    """정책 입력용 obs 정규화 — FixedNormalizeObs.observation과 *비트 동일*해야 한다.
    ⚠️ raw는 env가 준 float64 그대로 넣어라. float32로 미리 캐스팅하면 정규화 입력 정밀도가
       달라져 미로 궤적이 갈라지고 결정적 성공률이 eval_flip(88%)과 어긋난다(faithfulness 버그).
       mean/std는 float32(FixedNormalizeObs와 동일). 저장은 별도로 float32 캐스팅한다."""
    return np.clip((raw - mean) / (std + 1e-6), -CLIP, CLIP).astype(np.float32)


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ).decode().strip()
    except Exception:
        return "unknown"


def collect(cfg, model, obs_mean, obs_std, n_episodes, det_frac, seed0=20000):
    """n_episodes 롤아웃을 굴려 전이 버퍼(dict of lists)와 에피소드 통계를 반환."""
    env = build_env_raw(cfg)
    u = env.unwrapped
    n_det = int(round(det_frac * n_episodes))

    buf = {k: [] for k in (
        "states", "actions", "next_states", "rewards", "dones",
        "terminated", "truncated", "episode_ids", "episode_starts",
        "wp_idx", "wp_switch", "subgoal_xy", "action_mode")}
    ep_outcomes = []   # per-ep: ("reached"|"fell"|"timeout", steps, deterministic)

    for ep in range(n_episodes):
        deterministic = ep < n_det
        obs, info = env.reset(seed=seed0 + ep)
        wp_t = int(info.get("wp_idx", 0))
        done, step = False, 0
        term = trunc = False
        while not done:
            subgoal_t = np.asarray(u.goal_pos, dtype=np.float32).copy()  # 현재 obs의 서브목표
            policy_in = normalize(obs, obs_mean, obs_std)   # float64 obs로 정규화(=FixedNormalizeObs)
            raw_state = np.asarray(obs, dtype=np.float32)   # 저장은 float32
            action, _ = model.predict(policy_in, deterministic=deterministic)
            next_obs, reward, term, trunc, info = env.step(action)
            wp_next = int(info.get("wp_idx", wp_t))

            buf["states"].append(raw_state)
            buf["actions"].append(np.asarray(action, dtype=np.float32))
            buf["next_states"].append(np.asarray(next_obs, dtype=np.float32))
            buf["rewards"].append(np.float32(reward))
            buf["dones"].append(bool(term))                 # truncation은 done=0
            buf["terminated"].append(bool(term))
            buf["truncated"].append(bool(trunc))
            buf["episode_ids"].append(np.int32(ep))
            buf["episode_starts"].append(step == 0)
            buf["wp_idx"].append(np.int8(wp_t))
            buf["wp_switch"].append(wp_next != wp_t)        # 목표벡터 점프(비-Markov)
            buf["subgoal_xy"].append(subgoal_t)
            buf["action_mode"].append(np.int8(0 if deterministic else 1))

            obs, wp_t = next_obs, wp_next
            step += 1
            done = term or trunc

        reached = bool(info.get("reached_goal", False))
        outcome = "reached" if reached else ("timeout" if (trunc and not term) else "fell")
        ep_outcomes.append((outcome, step, deterministic))

    env.close()

    data = {
        "states": np.asarray(buf["states"], dtype=np.float32),
        "actions": np.asarray(buf["actions"], dtype=np.float32),
        "next_states": np.asarray(buf["next_states"], dtype=np.float32),
        "rewards": np.asarray(buf["rewards"], dtype=np.float32),
        "dones": np.asarray(buf["dones"], dtype=bool),
        "terminated": np.asarray(buf["terminated"], dtype=bool),
        "truncated": np.asarray(buf["truncated"], dtype=bool),
        "episode_ids": np.asarray(buf["episode_ids"], dtype=np.int32),
        "episode_starts": np.asarray(buf["episode_starts"], dtype=bool),
        "wp_idx": np.asarray(buf["wp_idx"], dtype=np.int8),
        "wp_switch": np.asarray(buf["wp_switch"], dtype=bool),
        "subgoal_xy": np.asarray(buf["subgoal_xy"], dtype=np.float32),
        "action_mode": np.asarray(buf["action_mode"], dtype=np.int8),
    }
    return data, ep_outcomes, n_det


def summarize(data, ep_outcomes, n_det, n_episodes):
    """수집 통계 dict(저장 provenance + 출력용)."""
    det = [o for o in ep_outcomes if o[2]]
    sto = [o for o in ep_outcomes if not o[2]]
    sr = lambda eps: (np.mean([o[0] == "reached" for o in eps]) if eps else float("nan"))
    lens = [o[1] for o in ep_outcomes]
    causes = {c: sum(o[0] == c for o in ep_outcomes) for c in ("reached", "fell", "timeout")}
    return dict(
        n_episodes=int(n_episodes), n_transitions=int(len(data["states"])),
        n_det=int(n_det), n_stoch=int(n_episodes - n_det),
        obs_dim=int(data["states"].shape[1]), action_dim=int(data["actions"].shape[1]),
        success_rate_det=round(float(sr(det)) * 100, 1),
        success_rate_stoch=round(float(sr(sto)) * 100, 1),
        success_rate_all=round(float(sr(ep_outcomes)) * 100, 1),
        mean_ep_len=round(float(np.mean(lens)), 1),
        median_ep_len=round(float(np.median(lens)), 1),
        termination=causes, n_wp_switches=int(data["wp_switch"].sum()),
    )


def assert_integrity(data):
    """저장 전 자기검증 — 잘못 배선되면 여기서 죽는다."""
    n = len(data["states"])
    assert data["states"].shape == (n, 109), data["states"].shape
    assert data["actions"].shape == (n, 8), data["actions"].shape
    assert data["next_states"].shape == (n, 109), data["next_states"].shape
    for k in ("states", "actions", "next_states", "rewards", "subgoal_xy"):
        assert np.isfinite(data[k]).all(), f"NaN/Inf in {k}"
    # 목표벡터 무결성: subgoal_xy(=wrapper 내부 goal_pos) == ant_xy + obs[107:109]
    err = np.abs(data["subgoal_xy"] - data["states"][:, 0:2] - data["states"][:, 107:109])
    assert err.max() < 1e-3, f"goal-vector mismatch max={err.max()}"
    # 에피소드 부기
    E = int(data["episode_ids"].max()) + 1
    assert int(data["episode_starts"].sum()) == E, "episode_starts != E"
    assert int((data["dones"] | data["truncated"]).sum()) == E, "한 에피소드는 정확히 한 번 끝나야"
    # 웨이포인트
    assert set(np.unique(data["wp_idx"]).tolist()) <= {0, 1, 2}, "wp_idx out of {0,1,2}"
    assert int(data["wp_switch"].sum()) <= 2 * E, "switch > 2/ep"


def verify_smoke(data, model, obs_mean, obs_std):
    """스모크 전용 추가 검증: raw 저장 여부 + 정책 충실도 라운드트립."""
    s = data["states"]
    norm = np.clip((s - obs_mean) / (obs_std + 1e-6), -CLIP, CLIP).astype(np.float32)
    assert not np.allclose(s, norm), "states가 정규화돼 저장됨(raw 아님)!"
    # 결정적 전이에서 저장 raw를 정규화→predict가 저장 action을 재현하는가
    det = data["action_mode"] == 0
    if det.any():
        idx = np.where(det)[0][:256]
        act, _ = model.predict(norm[idx], deterministic=True)
        mae = float(np.abs(act - data["actions"][idx]).mean())
        # 저장 raw가 float32라 재정규화 입력이 정책이 본 float64 경로와 ~1e-6 다름(단일스텝 open-loop라
        # 작음). 'raw 대신 정규화본을 저장'하는 대형 버그면 MAE가 폭발하므로 느슨한 경계로 충분.
        assert mae < 1e-3, f"policy round-trip MAE={mae} (raw obs가 정책 입력과 불일치)"
        print(f"  ✓ smoke 검증 통과 (raw 저장 + 정책 라운드트립 MAE={mae:.2e})")


def append_experiment_log(stats, meta):
    """EXPERIMENTS.md에 Phase 4.1 데이터셋 블록을 '추가만' 한다(mode 'a', 절대 재포맷 금지).
    PPO용 train_ppo.append_experiment_log는 model.logger/approx_kl 등 PPO 전용이라 재사용 X."""
    t = stats["termination"]
    block = (
        f'\n## Phase 4.1 — World-model rollout dataset ({meta["created_utc"]})\n'
        f'- run: {meta["run_name"]} | checkpoint: {meta["checkpoint"]}\n'
        f'- env: {meta["env_id"]} + {meta.get("env_wrapper")} | config: {meta["config"]}\n'
        f'- action mode: mixed det={stats["n_det"]}/stoch={stats["n_stoch"]} '
        f'(det_frac={meta["det_frac"]}) | seed0: {meta["seed0"]} | episodes: {stats["n_episodes"]}\n'
        f'- transitions: {stats["n_transitions"]} | obs_dim {stats["obs_dim"]} action_dim {stats["action_dim"]}\n'
        f'- success(reached_goal): det {stats["success_rate_det"]}% (cf. 로그 88%), '
        f'stoch {stats["success_rate_stoch"]}%, all {stats["success_rate_all"]}%\n'
        f'- ep_len: mean {stats["mean_ep_len"]} / median {stats["median_ep_len"]} / max 1000\n'
        f'- 종료: reached {t["reached"]}, fell/flip {t["fell"]}, timeout {t["timeout"]}\n'
        f'- waypoint switch(비-Markov): {stats["n_wp_switches"]} transitions flagged\n'
        f'- output: {meta["out"]} | norm(4.2 학습시 train-split): data/world_model_norm.npz\n'
        f'- git: {meta["git"]}\n'
        f'- 교훈: (직접 채우기)\n'
    )
    with open("EXPERIMENTS.md", "a", encoding="utf-8") as f:
        f.write(block)
    print("\n📝 EXPERIMENTS.md에 Phase 4.1 블록 추가됨 — '교훈' 칸 채워주세요")


def main():
    ap = argparse.ArgumentParser(description="Phase 4.1 — 월드모델용 rollout 수집")
    ap.add_argument("--config", default="configs/ppo_p3_maze_upright.yaml")
    ap.add_argument("--checkpoint", default="models/checkpoints_p3_maze_upright/ppo_final.zip")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--det-frac", type=float, default=0.5,
                    help="결정적 에피소드 비율(앞쪽). 나머지는 확률적.")
    ap.add_argument("--out", default="data/world_model_rollouts.npz")
    ap.add_argument("--seed0", type=int, default=20000)
    ap.add_argument("--run-name", default="p4_1_rollouts_v1")
    ap.add_argument("--no-log", action="store_true", help="EXPERIMENTS.md 추가 건너뜀")
    ap.add_argument("--smoke", action="store_true",
                    help="2 에피소드 + 무결성/라운드트립 검증, EXPERIMENTS 미기록, 임시 out")
    args = ap.parse_args()

    if args.smoke:
        if args.episodes == 1000:
            args.episodes = 2
        if args.out == "data/world_model_rollouts.npz":
            args.out = "data/_smoke_rollouts.npz"
        args.no_log = True

    cfg = yaml.safe_load(open(args.config))
    obs_mean, obs_std = load_norm_stats(cfg.get("norm_stats_path", "data/obs_norm_stats.npz"))
    obs_mean = obs_mean.astype(np.float32)   # FixedNormalizeObs와 동일 dtype
    obs_std = obs_std.astype(np.float32)
    model = PPO.load(args.checkpoint, device="cpu")
    model.set_random_seed(args.seed0)   # 확률적 롤아웃 재현성

    print(f"▶ 수집: {args.episodes} ep ({args.config} @ {args.checkpoint})", flush=True)
    data, ep_outcomes, n_det = collect(
        cfg, model, obs_mean, obs_std, args.episodes, args.det_frac, args.seed0)
    assert_integrity(data)
    if args.smoke:
        verify_smoke(data, model, obs_mean, obs_std)

    stats = summarize(data, ep_outcomes, n_det, args.episodes)
    meta = dict(
        run_name=args.run_name, checkpoint=args.checkpoint, config=args.config,
        env_id=cfg.get("env_id"), env_wrapper=cfg.get("env_wrapper"),
        det_frac=args.det_frac, seed0=args.seed0, out=args.out,
        created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        git=git_commit(),
        waypoints=(cfg.get("env_wrapper_kwargs") or {}).get("waypoints"),
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        meta_json=np.array(json.dumps({**meta, **stats}, ensure_ascii=False)),
        waypoints=np.asarray(meta["waypoints"] or [], dtype=np.float32),
        **data,
    )

    print("\n── 수집 완료 ──────────────────────")
    print(f"  transition           : {stats['n_transitions']}  (obs {stats['obs_dim']}, act {stats['action_dim']})")
    print(f"  성공률 det/stoch/all : {stats['success_rate_det']}% / "
          f"{stats['success_rate_stoch']}% / {stats['success_rate_all']}%")
    print(f"  평균/중앙 ep_len     : {stats['mean_ep_len']} / {stats['median_ep_len']}")
    print(f"  종료원인             : {stats['termination']}")
    print(f"  waypoint switch      : {stats['n_wp_switches']} (비-Markov flagged)")
    print(f"  저장                 : {args.out}")

    if not args.no_log:
        append_experiment_log(stats, meta)


if __name__ == "__main__":
    main()

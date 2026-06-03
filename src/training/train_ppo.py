"""
train_ppo.py — BC 정책을 warm-start 삼아 PPO로 강화학습 (3.1, 수정판).

검토 반영:
  ① 정규화 누락 수정: FixedNormalizeObs로 BC와 동일 고정 통계 정규화 (필수)
  ② log_std_init 낮춤: 기본 σ=1이 BC 행동을 덮는 문제 해결
  ③ warm-start 검증 게이트: 학습 '전' 결정적 평가로 성공률 ~5% 확인
  ④ W&B 집계 로깅: 매 스텝 대신 10k 요약 시점 1회
  ⑤ dummy/subproc 선택 (Mac 안전)
"""

import os
os.environ.setdefault("MUJOCO_GL", "glfw")

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import src.envs                              # AntMaze-v0 등록

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    SubprocVecEnv, DummyVecEnv, VecMonitor, VecNormalize,
)
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from src.imitation.behavior_cloning import BCPolicy
from src.envs.wrappers import FixedNormalizeObs


def load_norm_stats(path):
    """obs_norm_stats.npz → (mean, std). BC가 학습 때 쓴 고정 통계."""
    data = np.load(path)
    return data["obs_mean"], data["obs_std"]


def linear_schedule(initial_value):
    """SB3 lr 스케줄: progress_remaining(1→0)에 비례해 lr을 0까지 선형 감쇠.
    학습 후반 advantage가 뾰족해질 때 정책 step 크기를 자동으로 줄여 발산을 막는다."""
    def schedule(progress_remaining):
        return progress_remaining * initial_value
    return schedule


def apply_env_wrapper(env, env_wrapper, env_wrapper_kwargs=None, seed=None):
    """Stage-1 래퍼(random_goal/waypoint_follow)를 이름으로 씌운다. None이면 그대로.
    ⚠️ FixedNormalizeObs '안쪽'에서 적용해야 한다(래퍼가 raw obs/goal을 바꾼 뒤 정규화)."""
    if not env_wrapper:
        return env
    kw = dict(env_wrapper_kwargs or {})
    if env_wrapper == "random_goal":
        from src.envs.waypoint_follower import RandomGoalDirection
        return RandomGoalDirection(
            env, dist_min=kw.get("dist_min", 2.0), dist_max=kw.get("dist_max", 6.5),
            box=tuple(kw["box"]) if "box" in kw else (-3.4, 3.4, -1.4, 7.4),
            seed=seed)
    if env_wrapper == "waypoint_follow":
        from src.envs.waypoint_follower import WaypointFollower
        return WaypointFollower(
            env, waypoints=kw.get("waypoints"), wp_reach=kw.get("wp_reach", 0.8),
            subgoal_bonus=kw.get("subgoal_bonus", 10.0),
            final_bonus=kw.get("final_bonus", 50.0),
            goal_radius=kw.get("goal_radius", 1.0))   # 미로 fine-tune: 합리적 반경(1.5) 허용
    if env_wrapper == "random_sequence":
        from src.envs.waypoint_follower import RandomWaypointSequence
        return RandomWaypointSequence(
            env, max_goals=kw.get("max_goals", 5),
            dist_range=(kw.get("dist_min", 2.0), kw.get("dist_max", 4.0)),
            max_turn_angle=kw.get("max_turn_angle"),   # None=전방향(기존). 각도 커리큘럼용.
            seed=seed)
    raise ValueError(f"unknown env_wrapper: {env_wrapper}")


def make_env(seed, obs_mean, obs_std, env_id="AntMaze-v0", env_kwargs=None,
             env_wrapper=None, env_wrapper_kwargs=None):
    """병렬 env 1개 팩토리. BC와 동일 정규화 래퍼를 씌운다.
    env_id/env_kwargs로 미로(AntMaze-v0)/빈평지(AntMazeOpen-v0)/웨이포인트(AntMazeWaypoint-v0)를
    선택하고, env_wrapper로 Stage-1 래퍼를 (정규화 안쪽에) 끼운다."""
    env_kwargs = env_kwargs or {}
    def _init():
        env = gym.make(env_id, **env_kwargs)
        env = apply_env_wrapper(env, env_wrapper, env_wrapper_kwargs, seed=seed)
        env = FixedNormalizeObs(env, obs_mean, obs_std)
        return env
    return _init


def transfer_bc_weights(ppo_model, bc_path, device):
    """
    BC 정책 가중치를 PPO 정책으로 복사 (warm start).

      BC.net Linear[0,2,4] (은닉 3개) → PPO mlp_extractor.policy_net
      BC.net Linear[6]     (출력)     → PPO action_net (가우시안 평균 헤드)
      BC.net Tanh          버림 (PPO는 [-1,1] 클리핑으로 근사)
      log_std는 안 건드림 (policy_kwargs의 log_std_init로 이미 낮춤)

    SB3 2.8.0에서 경로/shape 실측 확인됨. shape 안 맞으면 건너뛰고 경고.
    """
    bc = BCPolicy()
    bc.load_state_dict(torch.load(bc_path, map_location=device))
    bc_linears = [m for m in bc.net if isinstance(m, nn.Linear)]
    # → [Linear(109,256), Linear(256,256), Linear(256,256), Linear(256,8)]

    policy = ppo_model.policy
    mlp_layers = [m for m in policy.mlp_extractor.policy_net
                  if isinstance(m, nn.Linear)]

    ok = True
    for bc_layer, sb3_layer in zip(bc_linears[:3], mlp_layers):
        if bc_layer.weight.shape == sb3_layer.weight.shape:
            sb3_layer.weight.data.copy_(bc_layer.weight.data)
            sb3_layer.bias.data.copy_(bc_layer.bias.data)
        else:
            ok = False
            print(f"⚠️ 은닉층 shape 불일치: "
                  f"{tuple(bc_layer.weight.shape)} vs {tuple(sb3_layer.weight.shape)}")

    if bc_linears[3].weight.shape == policy.action_net.weight.shape:
        policy.action_net.weight.data.copy_(bc_linears[3].weight.data)
        policy.action_net.bias.data.copy_(bc_linears[3].bias.data)
    else:
        ok = False
        print(f"⚠️ 출력층 shape 불일치: "
              f"{tuple(bc_linears[3].weight.shape)} vs {tuple(policy.action_net.weight.shape)}")

    print("✅ warm-start: BC 가중치 복사 완료" if ok
          else "⚠️ warm-start 일부 실패 — 위 경고 확인")
    return ok


def evaluate_warmstart(model, obs_mean, obs_std, n_episodes,
                       env_id="AntMaze-v0", env_kwargs=None,
                       env_wrapper=None, env_wrapper_kwargs=None, seed=10000):
    """
    학습 '전' 검증 게이트: warm-start된 정책을 결정적으로 평가.
    성공률이 BC 수준(~5%)이면 성공, ~0%면 정규화/매핑 실패.
    env_id/env_kwargs/env_wrapper는 학습과 동일하게 줘야 한다.
    """
    env = gym.make(env_id, **(env_kwargs or {}))
    env = apply_env_wrapper(env, env_wrapper, env_wrapper_kwargs, seed=seed)
    env = FixedNormalizeObs(env, obs_mean, obs_std)
    successes = 0
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        if info.get("reached_goal", False):
            successes += 1
    env.close()
    return successes / n_episodes


def final_eval(model, obs_mean, obs_std, env_id, env_kwargs=None,
               env_wrapper=None, env_wrapper_kwargs=None,
               n_episodes=100, seed=20000, flip_thresh=0.0):
    """학습 '후' 결정적 평가(권위 게이트).
    반환 (성공률, 평균 ep_len, flip_rate, flip_term_rate, mean_min_up_z).
    통과 기준 = 성공률 ≥기준 AND 평균 ep_len ≪ 1000. info['reached_goal'] 기준
    (waypoint_follow 모드에선 래퍼가 '최종 목표' 도달로 덮어씀).
    flip 지표 = 자세종료 효과 측정: up_z<flip_thresh로 뒤집힘 정의(eval_flip과 동일)."""
    env = gym.make(env_id, **(env_kwargs or {}))
    env = apply_env_wrapper(env, env_wrapper, env_wrapper_kwargs, seed=seed)
    env = FixedNormalizeObs(env, obs_mean, obs_std)
    u = env.unwrapped
    successes, lengths = 0, []
    flipped_eps, flip_term_eps, min_upz_per_ep = 0, 0, []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done, steps, ep_min_upz, ever_flipped = False, 0, 1.0, False
        terminated = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            uz = info.get("up_z")
            if uz is None:
                uz = u._up_z()           # 구 체크포인트(up_z info 없음) 폴백
            ep_min_upz = min(ep_min_upz, uz)
            ever_flipped = ever_flipped or (uz < flip_thresh)
            steps += 1
            done = terminated or truncated
        lengths.append(steps)
        min_upz_per_ep.append(ep_min_upz)
        reached = bool(info.get("reached_goal", False))
        if ever_flipped:
            flipped_eps += 1
        if terminated and not reached and ep_min_upz < flip_thresh:
            flip_term_eps += 1           # 뒤집혀서 종료(자세종료가 실제 발화)
        if reached:
            successes += 1
    env.close()
    return (successes / n_episodes, float(np.mean(lengths)),
            flipped_eps / n_episodes, flip_term_eps / n_episodes,
            float(np.mean(min_upz_per_ep)))

def verify_warmstart_match(model, bc_path, obs_mean, obs_std,
                           demo_path="data/expert_demos.npz", n=500):
    """
    warm-start 정확성을 '결정적으로' 검증 (성공률보다 신뢰도 높음).
    BC와 warm-start된 PPO가 같은 관측에서 거의 같은 행동을 내면 OK.
    반환: 행동 평균절대오차(MAE). 작을수록(예: <0.15) 정상.
    """
    bc = BCPolicy()
    bc.load_state_dict(torch.load(bc_path, map_location="cpu"))
    bc.eval()
    obs_raw = np.load(demo_path)["observations"]
    step = max(1, len(obs_raw) // n)
    sample = obs_raw[::step][:n].astype(np.float32)
    with torch.no_grad():
        bc_act = bc(torch.as_tensor(sample)).numpy()           # 내부 정규화 + tanh
    norm = ((sample - obs_mean) / (obs_std + 1e-6)).astype(np.float32)
    ppo_act, _ = model.predict(norm, deterministic=True)        # wrapper 정규화 + 클리핑
    return float(np.abs(bc_act - ppo_act).mean())


class ProgressCallback(BaseCallback):
    """진행상황 추적 + 보상 분해 집계 로깅 (10k 스텝마다 1회)."""

    def __init__(self, use_wandb=False, flip_thresh=0.0, verbose=0):
        super().__init__(verbose)
        self.use_wandb = use_wandb
        self.flip_thresh = flip_thresh      # 자세종료 임계(env up_thresh와 동일하게)
        self.episode_count = 0
        self.recent_success = []
        self.recent_steps = []
        self.recent_n_reached = []          # sequence 래퍼의 에피소드당 도달 목표 수(goals/ep)
        self.last_print = 0
        self._rew_sum = {"speed": 0.0, "energy": 0.0, "collision": 0.0,
                         "stall": 0.0, "time": 0.0, "progress": 0.0, "up": 0.0}
        self._rew_n = 0
        # 자세 추적: env별 진행중 min up_z(에피소드 경계로 fold) + 윈도우 집계
        self._cur_min_upz = {}              # env_idx → 현재 에피소드 min up_z
        self._upz_min_sum = 0.0             # 종료된 에피소드들의 min up_z 합
        self._upz_ep_n = 0
        self._flip_term_count = 0           # 뒤집혀 종료(도달X)한 에피소드 수(윈도우)

    def _on_step(self):
        for i, info in enumerate(self.locals.get("infos", [])):
            # 자세: env별 진행중 min up_z 갱신 (매 스텝)
            uz = info.get("up_z")
            if uz is not None:
                self._cur_min_upz[i] = min(self._cur_min_upz.get(i, 1.0), uz)
            if "episode" in info:
                self.episode_count += 1
                self.recent_steps.append(info["episode"]["l"])
                self.recent_success.append(1 if info.get("reached_goal", False) else 0)
                self.recent_n_reached.append(info.get("n_reached", 0))   # 없으면 0(타 stage 호환)
                self.recent_success = self.recent_success[-20:]
                self.recent_steps = self.recent_steps[-20:]
                self.recent_n_reached = self.recent_n_reached[-20:]
                # 자세: 에피소드 min up_z fold + 뒤집혀 종료(도달X) 카운트.
                #   새 env에선 flip 즉시 종료되므로 (도달X & min<thresh) ≈ flip-종료.
                emin = self._cur_min_upz.get(i, 1.0)
                self._upz_min_sum += emin
                self._upz_ep_n += 1
                if (not info.get("reached_goal", False)) and emin < self.flip_thresh:
                    self._flip_term_count += 1
                self._cur_min_upz[i] = 1.0
            # 보상 분해는 '매 스텝 로깅' 대신 누적만 (W&B 폭주 방지)
            if "speed_reward" in info:
                self._rew_sum["speed"] += info["speed_reward"]
                self._rew_sum["energy"] += info["energy_penalty"]
                self._rew_sum["collision"] += info["collision_penalty"]
                # Stage-0(direct) 전용 항 — 없으면 0 (미로 모드와 호환)
                self._rew_sum["stall"] += info.get("stall_penalty", 0.0)
                self._rew_sum["time"] += info.get("time_penalty", 0.0)
                self._rew_sum["progress"] += info.get("progress", 0.0)
                self._rew_sum["up"] += info.get("up_bonus", 0.0)
                self._rew_n += 1

        if self.num_timesteps - self.last_print >= 10000:
            self.last_print = self.num_timesteps
            sr = (np.mean(self.recent_success) * 100
                  if self.recent_success else 0.0)
            avg_steps = (np.mean(self.recent_steps)
                         if self.recent_steps else 0.0)
            goals_ep = (np.mean(self.recent_n_reached)
                        if self.recent_n_reached else 0.0)
            mean_min_upz = (self._upz_min_sum / self._upz_ep_n
                            if self._upz_ep_n else 1.0)
            print(f"  [{self.num_timesteps:>8} 스텝] "
                  f"에피소드 {self.episode_count} | "
                  f"최근 성공률 {sr:.1f}% | 평균 길이 {avg_steps:.0f} | "
                  f"goals/ep {goals_ep:.2f} | "
                  f"min_up_z {mean_min_upz:.2f} | flip종료 {self._flip_term_count}")
            if self.use_wandb and self._rew_n > 0:
                import wandb
                n = self._rew_n
                wandb.log({
                    "reward/speed": self._rew_sum["speed"] / n,
                    "reward/energy": self._rew_sum["energy"] / n,
                    "reward/collision": self._rew_sum["collision"] / n,
                    "reward/stall": self._rew_sum["stall"] / n,
                    "reward/time": self._rew_sum["time"] / n,
                    "reward/progress": self._rew_sum["progress"] / n,
                    "reward/up": self._rew_sum["up"] / n,           # 직립 보너스(+)
                    "rollout/success_rate": sr,
                    "rollout/ep_len_mean": avg_steps,   # Stage-0: 1000 아래로 떨어지는지 추적
                    "rollout/goals_per_ep": goals_ep,   # sequence: 붕괴(0.05)↔회복(→0.70) 직접 지표
                    "rollout/mean_min_up_z": mean_min_upz,  # →1 이면 전복 사라짐
                    "rollout/flip_term_count": self._flip_term_count,  # →0 이면 자세종료 발화 멈춤
                }, step=self.num_timesteps)
            self._rew_sum = {"speed": 0.0, "energy": 0.0, "collision": 0.0,
                             "stall": 0.0, "time": 0.0, "progress": 0.0, "up": 0.0}
            self._rew_n = 0
            self._upz_min_sum, self._upz_ep_n, self._flip_term_count = 0.0, 0, 0
        return True

def append_experiment_log(config, model, progress_cb, final_path, eval_result=None,
                          flip_result=None):
    """학습 종료 후 EXPERIMENTS.md에 결과 블록 자동 추가 (교훈/다음은 직접 채움).
    eval_result=(success_rate, mean_ep_len) 가 있으면 결정적 평가 + Stage-0 PASS/FAIL 기록.
    flip_result=(flip_rate, flip_term_rate, mean_min_up_z) 가 있으면 자세종료 효과 줄 추가."""
    from datetime import datetime
    L = model.logger.name_to_value
    sr = (np.mean(progress_cb.recent_success) * 100
          if progress_cb.recent_success else float("nan"))
    warm = (config.get("init_from") is None
            and os.path.exists(config["bc_policy_path"]))
    env_id = config.get("env_id", "AntMaze-v0")
    ekw = config.get("env_kwargs", {}) or {}
    eval_line = ""
    if eval_result is not None:
        ev_sr, ev_len = eval_result
        passed = (ev_sr >= 0.80 and ev_len < 1000)
        eval_line = (
            f'- 결정적평가(100ep): 성공률 {ev_sr*100:.1f}%, 평균 ep_len {ev_len:.0f} '
            f'→ Stage-0 {"PASS ✅" if passed else "FAIL ❌"}\n'
        )
    if flip_result is not None:
        fr, ftr, mmu = flip_result
        eval_line += (
            f'- 자세(100ep): flip_rate {fr*100:.1f}%, flip_term_rate {ftr*100:.1f}%, '
            f'mean_min_up_z {mmu:.3f}\n'
        )
    block = (
        f'\n## 실험 — {config.get("run_name","unnamed")} ({datetime.now():%Y-%m-%d %H:%M})\n'
        f'- env: {env_id} {ekw}, init_from={config.get("init_from")}\n'
        f'- 설정: warm-start={warm}, log_std_init={config["log_std_init"]}, '
        f'gamma={config.get("gamma")}, '
        f'target_kl={config.get("target_kl")}, total_timesteps={config["total_timesteps"]:,}\n'
        f'- 보상: goal_bonus={ekw.get("goal_bonus","(reg기본)")}, '
        f'stall={ekw.get("stall_penalty","(reg기본)")}, '
        f'time={ekw.get("time_penalty","(reg기본)")}\n'
        f'- 결과(학습말미 rolling20): 성공률 {sr:.1f}%, '
        f'approx_kl {L.get("train/approx_kl", float("nan")):.4f}, '
        f'std {L.get("train/std", float("nan")):.3f}, '
        f'clip_frac {L.get("train/clip_fraction", float("nan")):.2f}\n'
        f'{eval_line}'
        f'- 모델: {final_path}\n'
        f'- 교훈: (직접 채우기)\n'
        f'- 다음: (직접 채우기)\n'
    )
    with open("EXPERIMENTS.md", "a", encoding="utf-8") as f:
        f.write(block)
    print("\n📝 EXPERIMENTS.md에 결과 추가됨 — '교훈/다음' 칸 채워주세요")

def main():
    import argparse, yaml
    parser = argparse.ArgumentParser(description="AntMaze PPO (warm-start) 학습/스모크")
    parser.add_argument("--config", default="configs/ppo_config.yaml")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="total_timesteps 오버라이드 (스모크용)")
    parser.add_argument("--run-name", default=None, help="run_name 오버라이드")
    parser.add_argument("--no-wandb", action="store_true", help="W&B 끄기")
    parser.add_argument("--smoke", action="store_true",
                        help="스모크 실행: 기본 200k·wandb off·결과를 EXPERIMENTS.md/ppo_final에 안 남김")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    smoke = args.smoke
    if smoke:
        # 본 config는 건드리지 않고 메모리에서만 스모크 기본값 적용 (명시 플래그가 우선)
        config["total_timesteps"] = 200000
        config["use_wandb"] = False
        config["run_name"] = "200k_smoke_stabv2"
    if args.timesteps is not None:
        config["total_timesteps"] = args.timesteps
    if args.run_name is not None:
        config["run_name"] = args.run_name
    if args.no_wandb:
        config["use_wandb"] = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    # 0) BC 고정 정규화 통계 로드 (warm-start의 핵심)
    obs_mean, obs_std = load_norm_stats(config["norm_stats_path"])

    # env 선택: 미로(기본 AntMaze-v0) / 빈평지(AntMazeOpen-v0, Stage 0).
    #   env_kwargs(예: {goal_y: 4.0})로 등록 기본값을 per-run 덮어쓴다(거리 커리큘럼).
    env_id = config.get("env_id", "AntMaze-v0")
    env_kwargs = config.get("env_kwargs", {}) or {}
    env_wrapper = config.get("env_wrapper")           # Stage 1: random_goal / waypoint_follow
    env_wrapper_kwargs = config.get("env_wrapper_kwargs", {}) or {}
    print(f"env: {env_id}  env_kwargs={env_kwargs}  env_wrapper={env_wrapper}")

    # 1) 병렬 환경 (정규화 래퍼 포함)
    env_fns = [make_env(seed=i, obs_mean=obs_mean, obs_std=obs_std,
                        env_id=env_id, env_kwargs=env_kwargs,
                        env_wrapper=env_wrapper, env_wrapper_kwargs=env_wrapper_kwargs)
               for i in range(config["n_envs"])]
    env = (DummyVecEnv(env_fns) if config.get("vec_env") == "dummy"
           else SubprocVecEnv(env_fns))
    env = VecMonitor(env)                       # ★ VecNormalize '안쪽' → ep_rew_mean은 원시값 유지(스모크 게이트용)
    # 보상만 정규화: 원시 -800 보상을 O(1)로 → value_loss/advantage 스케일 안정 (요인#2).
    # norm_obs=False 필수 — obs는 FixedNormalizeObs 전용(켜면 이중정규화로 warm-start 파괴).
    if config.get("norm_reward", False):
        env = VecNormalize(
            env,
            norm_obs=False,
            norm_reward=True,
            clip_reward=config.get("clip_reward", 10.0),
            gamma=config["gamma"],
        )

    # 2) W&B (옵션)
    if config["use_wandb"]:
        import wandb
        wandb.init(project=config["wandb_project"],
                   name=config.get("run_name"),     # ★ 실험 이름
                   config=config)


    # 3) PPO 생성
    #    init_from 이 있으면 = 커리큘럼 다음 단계: 이전(더 쉬운 거리) 체크포인트를 이어학습.
    #    없으면 = 새 PPO 생성 후, BC 정책이 있으면 warm-start (없으면 scratch).
    init_from = config.get("init_from")
    if init_from:
        print(f"\n🔗 커리큘럼 이어학습: {init_from} 로드 (BC warm-start 건너뜀)")
        model = PPO.load(init_from, env=env, device=device,
                         custom_objects={"learning_rate": linear_schedule(config["learning_rate"])})
        # 자세종료 fine-tune: 체크포인트 log_std(σ~0.53)가 너무 커 탐색이 매 에피소드 전복시킴
        #   → 자세종료가 78% 에피소드를 즉시 끝내 학습 굶김(측정: σ0.53서 flip 93%).
        #   override로 탐색 노이즈를 낮춰(σ~0.22) 정책이 깨끗한 보행 근처에 머물게 함
        #   → 전복 33%로 감소, 대부분 에피소드 생존 → 신호 충분 → 직립 보너스+종료가 잔여 전복 제거.
        ovr = config.get("override_log_std")
        if ovr is not None:
            with torch.no_grad():
                model.policy.log_std.fill_(float(ovr))
            print(f"   log_std 강제설정: {ovr} (σ≈{np.exp(float(ovr)):.3f}) — 탐색 노이즈 축소(전복 굶김 방지)")
    else:
        policy_kwargs = dict(
            net_arch=[256, 256, 256],
            activation_fn=nn.ReLU,
            log_std_init=config["log_std_init"],   # 기본 0(σ=1) → BC 보존 위해 낮춤
        )
        model = PPO(
            "MlpPolicy", env,
            learning_rate=linear_schedule(config["learning_rate"]),  # 1e-4 → 0 선형감쇠 (요인#3)
            ent_coef=config.get("ent_coef", 0.0),    # ★ σ가 숨쉴 공간 → "평균 전용 KL 폭발" 차단 (요인#1)
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            gamma=config["gamma"],
            clip_range=config["clip_range"],
            n_epochs=config["n_epochs"],
            target_kl=config["target_kl"],          # ★ warm-start 발산 방지 (핵심, 절대 풀지 말 것)
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=1,
        )

        # 4) warm-start + 검증 게이트
        if os.path.exists(config["bc_policy_path"]):
            transfer_bc_weights(model, config["bc_policy_path"], device)

            # (a) 신뢰도 높은 검증: BC와 행동이 일치하는가 (결정적, 노이즈 없음)
            mae = verify_warmstart_match(model, config["bc_policy_path"], obs_mean, obs_std)
            flag = "✅ 정상" if mae < 0.15 else "❌ 의심 (정규화/매핑 확인 필요)"
            print(f"\nwarm-start 행동 일치 검증: BC와 행동 MAE = {mae:.4f}  {flag}")

            # (b) 참고용 성공률 (BC ~5%라 0%여도 정상 — 게이트로 쓰지 말 것)
            sr = evaluate_warmstart(model, obs_mean, obs_std,
                                    config["warmstart_eval_episodes"],
                                    env_id=env_id, env_kwargs=env_kwargs,
                                    env_wrapper=env_wrapper,
                                    env_wrapper_kwargs=env_wrapper_kwargs)
            print(f"  (참고) warm-start 직후 성공률: {sr*100:.1f}%  — BC 수준이라 0%여도 정상")
        else:
            print("⚠️ BC 정책 없음(scratch) — random 초기화로 학습")

    # 5) 콜백
    checkpoint_cb = CheckpointCallback(
        save_freq=max(config["checkpoint_freq"] // config["n_envs"], 1),
        save_path=config["checkpoint_dir"],
        name_prefix="ppo_antmaze",
    )
    progress_cb = ProgressCallback(use_wandb=config["use_wandb"],
                                   flip_thresh=float(env_kwargs.get("up_thresh", 0.0)))
    callbacks = [progress_cb] if smoke else [checkpoint_cb, progress_cb]

    # 6) 학습
    tag = "스모크" if smoke else "풀런"
    print(f"\nPPO {tag} 시작 (총 {config['total_timesteps']:,} 스텝)...\n")
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=callbacks,
    )

    # 7) 저장
    is_vecnorm = isinstance(env, VecNormalize)
    if smoke:
        final_path = os.path.join(config["checkpoint_dir"], "ppo_smoke.zip")
        model.save(final_path)
        if is_vecnorm:
            env.save(os.path.join(config["checkpoint_dir"], "vecnormalize_smoke.pkl"))
        print(f"\n✅ 스모크 모델 저장: {final_path}  (EXPERIMENTS.md/ppo_final 미기록)")
    else:
        final_path = os.path.join(config["checkpoint_dir"], "ppo_final.zip")
        model.save(final_path)
        if is_vecnorm:
            # 보상 정규화 통계 — 재개/평가 시 필요 (평가는 training=False로 로드)
            env.save(os.path.join(config["checkpoint_dir"], "vecnormalize.pkl"))
        print(f"\n✅ 최종 모델 저장: {final_path}")

        # 8) 결정적 100ep 평가(권위 게이트): 성공률 + 평균 ep_len → Stage-0 PASS/FAIL
        #    + 자세종료 효과 측정용 flip 지표(flip_rate/flip_term_rate/mean_min_up_z).
        flip_thresh = float((env_kwargs or {}).get("up_thresh", 0.0))
        ev_sr, ev_len, flip_rate, flip_term_rate, mean_min_upz = final_eval(
            model, obs_mean, obs_std,
            env_id=env_id, env_kwargs=env_kwargs,
            env_wrapper=env_wrapper, env_wrapper_kwargs=env_wrapper_kwargs,
            n_episodes=100, flip_thresh=flip_thresh)
        passed = (ev_sr >= 0.80 and ev_len < 1000)
        print(f"\n📊 결정적평가(100ep): 성공률 {ev_sr*100:.1f}% | 평균 ep_len {ev_len:.0f} "
              f"→ Stage-0 {'PASS ✅' if passed else 'FAIL ❌'}")
        print(f"   자세: flip_rate {flip_rate*100:.1f}% | flip_term_rate {flip_term_rate*100:.1f}% "
              f"| mean_min_up_z {mean_min_upz:.3f}  (flip_thresh={flip_thresh})")
        if config["use_wandb"]:
            wandb.log({"eval/success_rate": ev_sr * 100, "eval/ep_len_mean": ev_len,
                       "eval/flip_rate": flip_rate * 100,
                       "eval/flip_term_rate": flip_term_rate * 100,
                       "eval/mean_min_up_z": mean_min_upz})
        append_experiment_log(config, model, progress_cb, final_path,
                              eval_result=(ev_sr, ev_len),
                              flip_result=(flip_rate, flip_term_rate, mean_min_upz))

    env.close()
    if config["use_wandb"]:
        wandb.finish()


if __name__ == "__main__":
    main()

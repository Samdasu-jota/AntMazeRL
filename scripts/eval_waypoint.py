"""
Stage 1 — 웨이포인트 따라가기 평가 (+ top-down 영상).

검증된 보행 정책이 '서브목표(웨이포인트)'를 받아 미로(기둥)를 도는지 결정적으로 측정.
서브목표는 하드코딩 WAYPOINTS(기본) 또는 A*(--use-astar)가 생성.

사용:
  python -m scripts.eval_waypoint --checkpoint models/checkpoints_stage0_scratch_6m/ppo_final.zip \
      --episodes 100 [--video outputs/videos/stage1.mp4] [--use-astar]
"""
import os
os.environ.setdefault("MUJOCO_GL", "glfw")

import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

import src.envs  # noqa: F401  (AntMazeWaypoint-v0 등록)
from src.envs.waypoint_follower import WaypointFollower
from src.envs.wrappers import FixedNormalizeObs
from src.training.train_ppo import load_norm_stats

# 위에서 내려다보는 카메라 (메모리: camera_id=-1 + default_camera_config 동시 지정 필수)
TOPDOWN_CAM = {
    "trackbodyid": -1, "distance": 12.0, "elevation": -90.0,
    "azimuth": 90.0, "lookat": np.array([0.0, 3.0, 0.0]),
}


def _waypoints(use_astar):
    if not use_astar:
        return None  # WaypointFollower 기본 = 하드코딩 WAYPOINTS
    from src.planning.astar import plan
    wps = plan(start=(0.0, 0.0), goal=(0.0, 6.0))
    print("A* 웨이포인트:", [np.round(w, 2).tolist() for w in wps])
    return wps


def _make(obs_mean, obs_std, waypoints, render=False):
    mk = (dict(render_mode="rgb_array", camera_id=-1,
               default_camera_config=TOPDOWN_CAM) if render else {})
    env = gym.make("AntMazeWaypoint-v0", **mk)
    env = WaypointFollower(env, waypoints=waypoints)
    return FixedNormalizeObs(env, obs_mean, obs_std)


def run_eval(model, env, n_episodes, seed0=20000):
    succ, lengths, max_wp = 0, [], []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed0 + ep)
        done, steps, mwp = False, 0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            steps += 1
            mwp = max(mwp, info.get("wp_idx", 0))
            done = term or trunc
        lengths.append(steps)
        max_wp.append(mwp)
        if info.get("reached_goal", False):
            succ += 1
    return succ / n_episodes, float(np.mean(lengths)), float(np.mean(max_wp))


def render_episode(model, env, seed=20000):
    frames = []
    obs, info = env.reset(seed=seed)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        frames.append(env.render())
        done = term or trunc
    return frames, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--video", default=None)
    ap.add_argument("--use-astar", action="store_true")
    ap.add_argument("--norm-stats", default="data/obs_norm_stats.npz")
    args = ap.parse_args()

    obs_mean, obs_std = load_norm_stats(args.norm_stats)
    model = PPO.load(args.checkpoint, device="cpu")
    waypoints = _waypoints(args.use_astar)

    env = _make(obs_mean, obs_std, waypoints, render=False)
    sr, mean_len, mean_wp = run_eval(model, env, args.episodes)
    env.close()
    print(f"\n웨이포인트 따라가기: 성공률 {sr*100:.1f}% | 평균 ep_len {mean_len:.0f} "
          f"| 평균 도달 wp_idx {mean_wp:.2f}  (0=첫 +x 다리서 막힘, 2=최종 도달)")

    if args.video:
        renv = _make(obs_mean, obs_std, waypoints, render=True)
        frames, info = render_episode(model, renv, seed=20000)
        renv.close()
        import imageio.v2 as imageio
        os.makedirs(os.path.dirname(args.video) or ".", exist_ok=True)
        imageio.mimsave(args.video, frames, fps=30, macro_block_size=None)
        print(f"🎥 영상 저장: {args.video}  (이 에피소드 최종도달={info.get('reached_goal')})")


if __name__ == "__main__":
    main()

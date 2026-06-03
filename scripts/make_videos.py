"""
Stage 1 진단 영상 2종 (top-down):
  1) 잘하는 것  — Stage 0 87% 워커가 빈 평지에서 목표까지 직진 (AntMazeOpen-v0, goal_y=6)
  2) 못하는 것  — 미로에서 leg-1(+x)은 가다 leg-2(+y 턴)에서 막힘 (AntMazeWaypoint-v0 + WaypointFollower)

영상2는 'wp_idx 1로 advance(=첫 +x 다리 도달) 후 최종 미도달(=+y 턴서 정체)'가 또렷한
시드를 비렌더 스캔으로 먼저 고른 뒤 그 에피소드만 렌더한다.

사용: venv/bin/python -m scripts.make_videos
"""
import os
os.environ.setdefault("MUJOCO_GL", "glfw")

import numpy as np
import gymnasium as gym
import imageio.v2 as imageio
from stable_baselines3 import PPO

import src.envs  # noqa: F401  (env 등록)
from src.envs.waypoint_follower import WaypointFollower
from src.envs.wrappers import FixedNormalizeObs
from src.training.train_ppo import load_norm_stats

# 위에서 내려다보는 카메라 (메모리: camera_id=-1 + default_camera_config 동시 지정 필수)
TOPDOWN_CAM = {
    "trackbodyid": -1, "distance": 12.0, "elevation": -90.0,
    "azimuth": 90.0, "lookat": np.array([0.0, 3.0, 0.0]),
}

OM, OSD = load_norm_stats("data/obs_norm_stats.npz")
WALKER_STAGE0 = "models/checkpoints_stage0_scratch_6m/ppo_final.zip"
WALKER_MAZE = "models/checkpoints_stage1_cmdfollow_turn30/ppo_final.zip"


def rollout(env, model, seed, render=False, max_frames=None):
    """1 에피소드 실행. render=True면 프레임 수집. 반환 (frames, info, max_wp_idx, steps)."""
    obs, info = env.reset(seed=seed)
    done, mwp, steps, frames = False, 0, 0, []
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(a)
        steps += 1
        mwp = max(mwp, info.get("wp_idx", 0))
        if render:
            frames.append(env.render())
            if max_frames and len(frames) >= max_frames:
                break
        done = term or trunc
    return frames, info, mwp, steps


def make_open_plane_walk(out="outputs/videos/01_stage0_평지보행_87%/평지_직진_87%워커.mp4"):
    """영상1: Stage 0 87% 워커가 빈 평지(6m)에서 목표까지 직진 — 성공 시드 골라 렌더."""
    model = PPO.load(WALKER_STAGE0, device="cpu")
    scan = FixedNormalizeObs(gym.make("AntMazeOpen-v0", goal_y=6.0), OM, OSD)
    pick = 0
    for s in range(25):
        _, info, _, steps = rollout(scan, model, seed=1000 + s)
        if info.get("reached_goal"):       # 성공 + 너무 길지 않은(깔끔한) 에피소드
            pick = 1000 + s
            break
    scan.close()
    renv = FixedNormalizeObs(
        gym.make("AntMazeOpen-v0", goal_y=6.0, render_mode="rgb_array",
                 camera_id=-1, default_camera_config=TOPDOWN_CAM), OM, OSD)
    frames, info, _, steps = rollout(renv, model, seed=pick, render=True)
    renv.close()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    imageio.mimsave(out, frames, fps=30, macro_block_size=None)
    print(f"🎥 영상1(평지 직진): {out}  seed={pick} reached={info.get('reached_goal')} "
          f"steps={steps}")


def make_maze_stuck(out="outputs/videos/02_stage1_미로정체_턴병목/평지워커_위쪽턴_막힘.mp4", max_frames=700):
    """영상2: 미로에서 leg-1(+x) 도달 후 leg-2(+y 턴)서 막힘 — wp_idx 1 정체 시드 골라 렌더."""
    model = PPO.load(WALKER_MAZE, device="cpu")
    scan = FixedNormalizeObs(WaypointFollower(gym.make("AntMazeWaypoint-v0")), OM, OSD)
    pick, pick_mwp = None, -1
    for s in range(80):
        _, info, mwp, _ = rollout(scan, model, seed=3000 + s)
        # 원하는 데모: 첫 +x 다리는 통과(mwp>=1)했지만 최종 미도달(=+y 턴서 정체)
        if mwp == 1 and not info.get("reached_goal"):
            pick = 3000 + s
            break
        if mwp > pick_mwp and not info.get("reached_goal"):
            pick, pick_mwp = 3000 + s, mwp     # 차선책 기억
    scan.close()
    # ⚠️ 렌더 env도 스캔과 동일하게 WaypointFollower로 감싸야 서브목표/wp_idx 로직이 동작
    renv = FixedNormalizeObs(
        WaypointFollower(gym.make("AntMazeWaypoint-v0", render_mode="rgb_array",
                                  camera_id=-1, default_camera_config=TOPDOWN_CAM)),
        OM, OSD)
    frames, info, mwp, steps = rollout(renv, model, seed=pick, render=True,
                                       max_frames=max_frames)
    renv.close()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    imageio.mimsave(out, frames, fps=30, macro_block_size=None)
    print(f"🎥 영상2(미로 막힘): {out}  seed={pick} max_wp_idx={mwp} "
          f"reached={info.get('reached_goal')} steps={steps} frames={len(frames)} "
          f"(wp_idx 1=첫 +x 다리 도달 후 +y 턴서 정체)")


if __name__ == "__main__":
    make_open_plane_walk()
    make_maze_stuck()

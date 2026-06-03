"""
축소 기둥(1/3) + A* 웨이포인트 미로 결과 영상 — honest spawn(0,0), goal_radius 1.0.
  영상1: 성공 (짧은 기둥을 A* gentle 경로로 우회해 완주)
  영상2: 실패 (어디서 막히는지)
사용: venv/bin/python -m scripts.make_maze_short_video
"""
import os
os.environ.setdefault("MUJOCO_GL", "glfw")

import numpy as np
import gymnasium as gym
import imageio.v2 as imageio
from stable_baselines3 import PPO

import src.envs  # noqa: F401
from src.envs.waypoint_follower import WaypointFollower
from src.envs.wrappers import FixedNormalizeObs
from src.training.train_ppo import load_norm_stats

TOPDOWN_CAM = {
    "trackbodyid": -1, "distance": 13.0, "elevation": -90.0,
    "azimuth": 90.0, "lookat": np.array([0.0, 3.0, 0.0]),
}
MODEL = "models/checkpoints_stage1_maze_short/ppo_final.zip"
WPS = [(-1.0, 1.4), (-1.0, 4.6), (0.0, 6.0)]   # A* (pillar_half_len=1.0)
OM, OSD = load_norm_stats("data/obs_norm_stats.npz")


def make_env(render=False):
    mk = (dict(render_mode="rgb_array", camera_id=-1, default_camera_config=TOPDOWN_CAM)
          if render else {})
    base = gym.make("AntMazeWaypoint-v0", pillar_half_len=1.0, start_xy=[0.0, 0.0], **mk)
    return FixedNormalizeObs(
        WaypointFollower(base, waypoints=[np.asarray(w, float) for w in WPS],
                         goal_radius=1.0), OM, OSD)


def rollout(env, model, seed, render=False, max_frames=None):
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


def main():
    model = PPO.load(MODEL, device="cpu")
    scan = make_env(render=False)
    succ_seed, fail_seed = None, None
    for s in range(120):
        _, info, mwp, _ = rollout(scan, model, seed=7000 + s)
        if info.get("reached_goal") and succ_seed is None:
            succ_seed = 7000 + s
        if (not info.get("reached_goal")) and mwp >= 1 and fail_seed is None:
            fail_seed = 7000 + s
        if succ_seed is not None and fail_seed is not None:
            break
    scan.close()
    os.makedirs("outputs/videos/04_stage1_축소기둥_58%", exist_ok=True)
    os.makedirs("outputs/videos/04_stage1_축소기둥_58%", exist_ok=True)

    if succ_seed is not None:
        renv = make_env(render=True)
        frames, info, mwp, steps = rollout(renv, model, seed=succ_seed, render=True)
        renv.close()
        imageio.mimsave("outputs/videos/04_stage1_축소기둥_58%/축소기둥미로_성공_58-65%.mp4", frames, fps=30, macro_block_size=None)
        print(f"🎥 성공: outputs/videos/04_stage1_축소기둥_58%/축소기둥미로_성공_58-65%.mp4  seed={succ_seed} steps={steps} (축소기둥 완주!)")
    else:
        print("⚠️ 성공 시드 못 찾음")

    if fail_seed is not None:
        renv = make_env(render=True)
        frames, info, mwp, steps = rollout(renv, model, seed=fail_seed, render=True, max_frames=700)
        renv.close()
        imageio.mimsave("outputs/videos/04_stage1_축소기둥_58%/축소기둥미로_실패.mp4", frames, fps=30, macro_block_size=None)
        print(f"🎥 실패: outputs/videos/04_stage1_축소기둥_58%/축소기둥미로_실패.mp4  seed={fail_seed} max_wp_idx={mwp} frames={len(frames)}")
    else:
        print("⚠️ 실패 시드 못 찾음(거의 다 성공이면 정상)")


if __name__ == "__main__":
    main()

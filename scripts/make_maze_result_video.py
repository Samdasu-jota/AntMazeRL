"""
미로 fine-tune(maze_rebalance) 결과 영상 — '우리가 테스트한 그 모델'을 눈으로 확인.
기하 수정 적용: 스폰 (0,-1) + goal_radius 1.5 + 하드코딩 웨이포인트.
  영상1: 성공 케이스 (미로 완주 — 기둥 우회해 목표 도달)
  영상2: 실패 케이스 (어디서 막히는지, 보통 wp_idx 1 정체)
사용: venv/bin/python -m scripts.make_maze_result_video
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
    "trackbodyid": -1, "distance": 12.5, "elevation": -90.0,
    "azimuth": 90.0, "lookat": np.array([1.0, 2.5, 0.0]),
}
MODEL = "models/checkpoints_stage1_maze_rebalance/ppo_final.zip"
OM, OSD = load_norm_stats("data/obs_norm_stats.npz")


def make_env(render=False):
    mk = (dict(render_mode="rgb_array", camera_id=-1, default_camera_config=TOPDOWN_CAM)
          if render else {})
    base = gym.make("AntMazeWaypoint-v0", start_xy=[0.0, -1.0], **mk)
    return FixedNormalizeObs(WaypointFollower(base, goal_radius=1.5), OM, OSD)


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
        _, info, mwp, _ = rollout(scan, model, seed=5000 + s)
        if info.get("reached_goal") and succ_seed is None:
            succ_seed = 5000 + s
        if (not info.get("reached_goal")) and mwp >= 1 and fail_seed is None:
            fail_seed = 5000 + s
        if succ_seed is not None and fail_seed is not None:
            break
    scan.close()
    os.makedirs("outputs/videos/03_stage1_rebalance_43%", exist_ok=True)
    os.makedirs("outputs/videos/03_stage1_rebalance_43%", exist_ok=True)

    if succ_seed is not None:
        renv = make_env(render=True)
        frames, info, mwp, steps = rollout(renv, model, seed=succ_seed, render=True)
        renv.close()
        imageio.mimsave("outputs/videos/03_stage1_rebalance_43%/리밸런스미로_성공_r1.5.mp4", frames, fps=30, macro_block_size=None)
        print(f"🎥 성공: outputs/videos/03_stage1_rebalance_43%/리밸런스미로_성공_r1.5.mp4  seed={succ_seed} steps={steps} (완주!)")
    else:
        print("⚠️ 성공 시드 못 찾음(120개 스캔)")

    if fail_seed is not None:
        renv = make_env(render=True)
        frames, info, mwp, steps = rollout(renv, model, seed=fail_seed, render=True, max_frames=700)
        renv.close()
        imageio.mimsave("outputs/videos/03_stage1_rebalance_43%/리밸런스미로_실패.mp4", frames, fps=30, macro_block_size=None)
        print(f"🎥 실패: outputs/videos/03_stage1_rebalance_43%/리밸런스미로_실패.mp4  seed={fail_seed} max_wp_idx={mwp} "
              f"frames={len(frames)} (보통 +y 턴/leg-1서 정체)")
    else:
        print("⚠️ 실패 시드 못 찾음")


if __name__ == "__main__":
    main()

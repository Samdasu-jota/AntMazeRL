"""
뒤집힘(전복) 증거 영상 — 옆/위 각도 추적 카메라로 'before(뒤집혀 기어감)' / 'after(똑바로 걸음)'를 렌더.

top-down 카메라는 위에서 봐서 전복을 못 보여준다(뒤집혀도 위에선 똑같이 보임).
이 스크립트는 몸통(torso)을 추적하는 살짝 위·옆 각도 카메라로 배가 위로 간 걸 또렷이 보여준다.
episode 선택: --want-inverted면 가장 많이 뒤집힌(min up_z 최저) 에피소드, 아니면 첫 성공(직립) 에피소드.
가장 뒤집힌 순간 프레임은 PNG로도 저장(카메라 각도 검증 + 단일 증거 샷).

사용:
  # before(미로 전복):
  venv/bin/python -m scripts.make_flip_video --config configs/ppo_stage1_maze_short.yaml \
      --checkpoint models/checkpoints_stage1_maze_short/ppo_final.zip \
      --out "outputs/videos/06_phase3_미로직립_88%/미로_before_뒤집힘_58%.mp4" --want-inverted
  # after(평지 직립): 학습 끝난 뒤 새 체크포인트로 동일 실행(--want-inverted 빼기)
"""
import os
os.environ.setdefault("MUJOCO_GL", "glfw")

import argparse
import numpy as np
import yaml
import gymnasium as gym
import imageio.v2 as imageio
from stable_baselines3 import PPO

import src.envs  # noqa: F401  (env 등록)
from src.envs.wrappers import FixedNormalizeObs
from src.training.train_ppo import load_norm_stats, apply_env_wrapper
from scripts.eval_flip import up_z_from_qpos

# 몸통 추적 + 살짝 위·옆 각도 → 전복(배가 위)이 또렷. (메모리: camera_id=-1 + default_camera_config 필수)
ANGLED_CAM = {
    "trackbodyid": 1,        # Ant torso (body 0=world, 1=torso)
    "distance": 6.0, "elevation": -20.0, "azimuth": 90.0,
    "lookat": np.array([0.0, 0.0, 0.0]),
}
# 위에서 수직(elevation -90)으로 본 미로 전체 → '주행 경로'(A* 따라가기)를 보여줌. 전복은 못 보여줌.
# (축소기둥 미로용; make_maze_short_video.TOPDOWN_CAM과 동일 세팅)
TOPDOWN_CAM = {
    "trackbodyid": -1, "distance": 13.0, "elevation": -90.0,
    "azimuth": 90.0, "lookat": np.array([0.0, 3.0, 0.0]),
}
CAMS = {"angled": ANGLED_CAM, "topdown": TOPDOWN_CAM}


def build_env(cfg, om, osd, seed=0, render=False, cam=ANGLED_CAM):
    mk = (dict(render_mode="rgb_array", camera_id=-1, default_camera_config=cam)
          if render else {})
    env = gym.make(cfg.get("env_id", "AntMaze-v0"), **(cfg.get("env_kwargs") or {}), **mk)
    env = apply_env_wrapper(env, cfg.get("env_wrapper"),
                            cfg.get("env_wrapper_kwargs") or {}, seed=seed)
    return FixedNormalizeObs(env, om, osd)


def rollout(env, model, seed, render=False, max_frames=None):
    """1 에피소드. 반환 (frames, upzs, info, min_uz). upzs[i]=프레임 i의 up_z."""
    u = env.unwrapped
    obs, info = env.reset(seed=seed)
    done, frames, upzs, min_uz = False, [], [], 1.0
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(a)
        uz = up_z_from_qpos(u.data.qpos)
        min_uz = min(min_uz, uz)
        if render:
            frames.append(env.render())
            upzs.append(uz)
            if max_frames and len(frames) >= max_frames:
                break
        done = term or trunc
    return frames, upzs, info, min_uz


def main():
    ap = argparse.ArgumentParser(description="전복 증거 영상 (각도 추적 카메라)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-scan", type=int, default=40, help="에피소드 선택용 비렌더 스캔 수")
    ap.add_argument("--seed0", type=int, default=20000)
    ap.add_argument("--max-frames", type=int, default=700)
    ap.add_argument("--want-inverted", action="store_true",
                    help="가장 많이 뒤집힌 에피소드 선택(기본=첫 직립 성공)")
    ap.add_argument("--cam", choices=("angled", "topdown"), default="angled",
                    help="angled=몸통추적 측면각(전복 증거, 기본) · topdown=수직 위(주행 경로)")
    ap.add_argument("--norm-stats", default=None)
    args = ap.parse_args()
    cam = CAMS[args.cam]

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    norm_path = args.norm_stats or cfg.get("norm_stats_path", "data/obs_norm_stats.npz")
    om, osd = load_norm_stats(norm_path)
    model = PPO.load(args.checkpoint, device="cpu")

    # 1) 비렌더 스캔으로 시드 선택
    scan = build_env(cfg, om, osd, render=False)
    best_seed, best_key = args.seed0, None
    for s in range(args.n_scan):
        seed = args.seed0 + s
        _, _, info, min_uz = rollout(scan, model, seed=seed)
        reached = bool(info.get("reached_goal", False))
        if args.want_inverted:
            key = -min_uz                       # 더 뒤집힐수록 큰 key
        else:
            key = (1.0 if reached else 0.0, min_uz)  # 성공 우선, 그다음 더 직립
        if best_key is None or key > best_key:
            best_key, best_seed = key, seed
    scan.close()

    # 2) 선택 시드 렌더
    renv = build_env(cfg, om, osd, seed=best_seed, render=True, cam=cam)
    frames, upzs, info, min_uz = rollout(renv, model, seed=best_seed, render=True,
                                         max_frames=args.max_frames)
    renv.close()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    imageio.mimsave(args.out, frames, fps=30, macro_block_size=None)

    # 3) 가장 뒤집힌 순간 프레임 → PNG (단일 증거 + 카메라 검증용; topdown은 전복이 안 보여 생략)
    if frames and args.cam == "angled":
        k = int(np.argmin(upzs))
        png = os.path.splitext(args.out)[0] + "_minupz.png"
        imageio.imwrite(png, frames[k])
        print(f"🖼  최저 up_z 프레임: {png}  (frame {k}/{len(frames)}, up_z={upzs[k]:.3f})")
    print(f"🎥 {args.out}  seed={best_seed} reached={info.get('reached_goal')} "
          f"min_up_z={min_uz:.3f} frames={len(frames)}")


if __name__ == "__main__":
    main()

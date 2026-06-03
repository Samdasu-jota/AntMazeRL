"""
preview_expert.py — 스크립트 전문가의 걸음새를 '눈으로' 확인하는 도구 (2.1).

대량 수집(collect_demos) 전에, 전문가가 실제로 걷는지/목표로 가는지 본다.
단계별로 점검:
  A (forward)  : 조향 없이 직진만 — 일단 '걷기'가 되는지 (걸음새 튜닝의 출발점)
  B (heading)  : 목표 한 점을 향해 방향 잡고 가는지
  C (waypoint) : 벽을 우회해 목표까지 도달하는지 (최종)

실행 (프로젝트 루트에서):
  python -m scripts.preview_expert        # 기본 = A단계(직진)
  python -m scripts.preview_expert B
  python -m scripts.preview_expert C

출력:
  - 콘솔: 에피소드별 순이동(Δx,Δy), 최종 목표거리, 도달 여부
  - 영상: outputs/videos/00_셋업_환경·전문가/전문가_<직진걷기|방향잡기|미로주행>.mp4  (개미 추적 시점)

걸음새가 이상하면 src/experts/scripted_expert.py 상단 상수
(FREQ / HIP_AMP / ANK_AMP / ANK_LAG / FORWARD_SIGN / 각 다리 ank_sign)를 조정한다.
A단계에서 '전진은 하는데 방향이 반대'면 FORWARD_SIGN을 뒤집고, '제자리에서 버둥'
이면 HIP_AMP/ANK_AMP/ANK_LAG를 조정한다.
"""

import os
os.environ.setdefault("MUJOCO_GL", "glfw")  # 검은화면이면 egl/osmesa로

import sys
import numpy as np
import imageio.v2 as imageio
import gymnasium as gym
import src.envs                              # ← AntMaze-v0 등록 (필수)
from src.envs.ant_maze_env import GOAL_POS, GOAL_RADIUS
from src.experts.scripted_expert import ScriptedExpert

STAGE_TO_MODE = {"A": "forward", "B": "heading", "C": "waypoint"}

# 개미 추적(follow) 자유 카메라 — test_env.py와 동일 패턴 (camera_id=-1과 함께 써야 적용)
FOLLOW_CAM = {
    "trackbodyid": -1,
    "distance": 8.0,
    "elevation": -30.0,
    "azimuth": 90.0,
    "lookat": np.array([0.0, 0.0, 0.0]),
}


def rollout(env, expert, seed, record=False):
    """한 에피소드를 전문가로 실행. record=True면 추적 시점 프레임을 모은다."""
    u = env.unwrapped
    obs, info = env.reset(seed=seed)
    expert.reset()

    cam = None
    frames = []
    if record:
        env.render()                         # 뷰어 생성 강제 → cam 접근 가능
        cam = u.mujoco_renderer.viewer.cam

    start_xy = obs[0:2].copy()
    last_info = info
    min_z = obs[2]
    done = False
    while not done:
        action = expert.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        last_info = info
        min_z = min(min_z, obs[2])           # 가장 낮았던 몸통 높이 (전복 여부)
        if record:
            cam.lookat[:] = u.data.qpos[:3]  # 개미를 따라가도록 갱신
            frames.append(env.render())
        done = terminated or truncated

    disp = obs[0:2] - start_xy
    return frames, last_info, disp, min_z


def main():
    stage = (sys.argv[1] if len(sys.argv) > 1 else "A").upper()
    mode = STAGE_TO_MODE.get(stage)
    if mode is None:
        print(f"알 수 없는 단계 '{stage}'. A / B / C 중 하나를 쓰세요.")
        sys.exit(1)

    print(f"── 전문가 미리보기: 단계 {stage} (mode='{mode}') ──")
    print(f"   목표 {tuple(GOAL_POS)} / 도달반경 {GOAL_RADIUS}m\n")

    # 깔끔한 관찰을 위해 노이즈 0으로 (수집 때는 노이즈 사용)
    expert = ScriptedExpert(mode=mode, noise_std=0.0)

    # 1) 통계: 5 에피소드 (렌더 없음 — 빠르게). min_z<0.4면 넘어진 것.
    env = gym.make("AntMaze-v0")
    for ep in range(5):
        _, info, disp, min_z = rollout(env, expert, seed=ep, record=False)
        toppled = "  ← 넘어짐" if min_z < 0.4 else ""
        print(f"  ep {ep}: 순이동 Δ=({disp[0]:+.2f}, {disp[1]:+.2f}) m | "
              f"최종 목표거리 {info['dist_to_goal']:.2f}m | "
              f"도달={'O' if info['reached_goal'] else 'X'} | "
              f"min_z={min_z:.2f}{toppled}")
    env.close()

    # 2) 영상: 1 에피소드 추적 시점 녹화
    env_v = gym.make("AntMaze-v0", render_mode="rgb_array",
                     camera_id=-1, default_camera_config=FOLLOW_CAM)
    frames, _, _, _ = rollout(env_v, expert, seed=0, record=True)
    env_v.close()

    _label = {"A": "전문가_직진걷기", "B": "전문가_방향잡기",
              "C": "전문가_미로주행"}.get(stage, f"전문가_{stage}")
    out = f"outputs/videos/00_셋업_환경·전문가/{_label}.mp4"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    imageio.mimsave(out, frames, fps=30, macro_block_size=None)
    print(f"\n✅ 영상 저장: {out} ({len(frames)} 프레임)")


if __name__ == "__main__":
    main()

"""
test_env.py — AntMazeEnv가 제대로 작동하는지 종합 확인 (1.3).

하는 일:
1. 환경 로드 + obs/action 공간 크기 검증(assert)
2. 랜덤 행동으로 3 에피소드 실행 + 보상 분해(속도/에너지/충돌) 출력
   - 각 에피소드마다 reward == speed - energy - collision 정합성 assert
3. 개미 경로(x,y)를 미로 위에 그려 PNG 저장 (matplotlib top-down 2D)
4. 한 에피소드를 두 카메라로 mp4 저장 — 같은 action 시퀀스를 리플레이해 동일 궤적을:
   - top-down  : 위에서 수직으로 내려다본 미로 전체 ("어디로 갔나")
   - follow    : 개미를 비스듬히 따라가는 추적 시점 ("어떻게 움직였나")

카메라 핵심: gymnasium은 기본적으로 ant.xml의 고정 'track' 카메라(camera_id=0, 지면 높이)를 써서
미로가 거의 안 보인다. camera_id=-1로 '자유 카메라'를 강제하면 default_camera_config가 적용된다.

실행: 프로젝트 루트에서  python -m scripts.test_env
"""

import os
os.environ.setdefault("MUJOCO_GL", "glfw")  # 검은화면이면 egl/osmesa로

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio                # v2 명시 → deprecation 경고 회피
import gymnasium as gym
import src.envs                             # ← AntMaze-v0 등록 (필수)
from src.envs.ant_maze_env import START_POS, GOAL_POS, GOAL_RADIUS


# ── 미로 벽 정보 (그림 전용; src/envs/ant_maze_env.py make_maze_xml의 geom을 하드코딩 복사) ──
# (중심x, 중심y, 절반너비hx, 절반높이hy). env XML이 바뀌면 이 그림은 조용히 어긋날 수 있음.
MAZE_WALLS = [
    (0,  3, 0.3, 3),   # 가운데 칸막이 (y: 0~6)
    (-4, 3, 0.3, 5),   # 왼쪽
    (4,  3, 0.3, 5),   # 오른쪽
    (0,  8, 4,   0.3), # 위
    (0, -2, 4,   0.3), # 아래
]

# 자유 카메라 설정 (camera_id=-1과 함께 써야 적용됨)
# top-down: 미로 중심을 수직으로 내려다보기
TOPDOWN_CAM = {
    "trackbodyid": -1,                    # 몸통 추적 끔 (고정 시점)
    "distance": 16.0,                     # 미로(약 9×11m)가 화면에 들어오는 거리
    "elevation": -90.0,                   # 정확히 수직 내려다보기
    "azimuth": 90.0,
    "lookat": np.array([0.0, 3.0, 0.0]),  # ndarray여야 cam.lookat[:]= 로 안전 적용됨
}
# follow: 비스듬히 위에서. lookat은 매 프레임 개미 위치로 갱신(아래 replay_actions follow=True)
FOLLOW_CAM = {
    "trackbodyid": -1,
    "distance": 8.0,
    "elevation": -30.0,
    "azimuth": 90.0,
    "lookat": np.array([0.0, 0.0, 0.0]),
}


def draw_maze(ax):
    """matplotlib 축(ax) 위에 미로 벽 + 시작/목표를 그린다 (벽 좌표 하드코딩)."""
    for cx, cy, hx, hy in MAZE_WALLS:
        # 사각형 왼쪽아래 모서리 = 중심 - 절반크기
        rect = plt.Rectangle(
            (cx - hx, cy - hy), 2 * hx, 2 * hy,
            color="gray", alpha=0.7
        )
        ax.add_patch(rect)
    # 시작점(초록), 목표점(빨강 별 + 도달반경 원)
    ax.plot(*START_POS, "go", markersize=12, label="Start")
    ax.plot(*GOAL_POS, "r*", markersize=18, label="Goal")
    goal_circle = plt.Circle(
        GOAL_POS, GOAL_RADIUS, color="red", fill=False, linestyle="--", alpha=0.5
    )
    ax.add_patch(goal_circle)


def run_episode(env, seed=None):
    """
    한 에피소드를 랜덤 행동으로 실행 (렌더 없음).
    seed를 주면 reset(seed)로 초기상태 고정 → actions 리플레이가 재현 가능.
    반환: path(np.array[(x,y)...]), totals(dict), actions(list)
    """
    obs, info = env.reset(seed=seed)
    path = []
    actions = []
    totals = {"speed": 0.0, "energy": 0.0, "collision": 0.0, "reward": 0.0}

    done = False
    while not done:
        action = env.action_space.sample()                 # 랜덤 행동
        obs, reward, terminated, truncated, info = env.step(action)
        actions.append(action)

        # obs 끝 2개 = 목표 상대벡터(GOAL_POS - ant_xy). 현재 위치 = 목표 - 상대벡터
        ant_xy = GOAL_POS - obs[-2:]
        path.append(ant_xy.copy())

        totals["speed"]     += info["speed_reward"]
        totals["energy"]    += info["energy_penalty"]
        totals["collision"] += info["collision_penalty"]
        totals["reward"]    += reward

        done = terminated or truncated

    return np.array(path), totals, actions


def replay_actions(env, actions, seed, follow=False):
    """
    저장된 action 시퀀스를 같은 seed로 리플레이하며 프레임 녹화 (카메라 각도만 다름).
    MuJoCo는 결정론적이라 (seed, actions)가 같으면 궤적이 동일하다.
    follow=True면 매 프레임 자유 카메라의 lookat을 개미 위치로 옮겨 추적 시점을 만든다.
    """
    u = env.unwrapped
    env.reset(seed=seed)

    cam = None
    if follow:
        env.render()                       # 뷰어 생성 강제 → viewer.cam 접근 가능
        cam = u.mujoco_renderer.viewer.cam

    frames = []
    for action in actions:
        _, _, terminated, truncated, _ = env.step(action)
        if follow:
            cam.lookat[:] = u.data.qpos[:3]  # 개미(몸통 x,y,z)를 따라가도록 갱신
        frames.append(env.render())
        if terminated or truncated:
            break
    return frames


def main():
    # ── 1) 환경 로드 + 공간 크기 검증 ──────────────────
    env = gym.make("AntMaze-v0", render_mode="rgb_array")
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape, \
        f"obs shape {obs.shape} != space {env.observation_space.shape}"
    assert env.action_space.shape == (8,), \
        f"action space {env.action_space.shape} != (8,)"
    print(f"✅ 공간 검증 통과: obs={obs.shape}, action={env.action_space.shape}\n")

    # ── 2) 3 에피소드 실행 + 보상 분해 출력 + 정합성 체크 ──
    all_paths = []
    for ep in range(3):
        path, totals, _ = run_episode(env)
        all_paths.append(path)

        # 총보상 = 속도 - 에너지 - 충돌 (info의 페널티는 양수 크기)
        recomputed = totals["speed"] - totals["energy"] - totals["collision"]
        assert abs(totals["reward"] - recomputed) < 1e-6, \
            f"보상 정합성 실패: total={totals['reward']} vs {recomputed}"

        print(f"── 에피소드 {ep + 1} ({len(path)} 스텝) ──")
        print(f"   속도 보상 합    : {totals['speed']:8.2f}")
        print(f"   에너지 페널티 합: {totals['energy']:8.2f}")
        print(f"   충돌 페널티 합  : {totals['collision']:8.2f}")
        print(f"   총 보상         : {totals['reward']:8.2f}\n")

    # ── 3) 경로 그림 저장 ──────────────────────────────
    os.makedirs("outputs/images", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 9))
    draw_maze(ax)
    colors = ["blue", "orange", "green"]
    for i, path in enumerate(all_paths):
        if len(path) > 0:
            ax.plot(path[:, 0], path[:, 1], color=colors[i],
                    alpha=0.7, label=f"Episode {i + 1}")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-3, 9)
    ax.set_aspect("equal")          # x,y 비율 1:1 (미로 안 찌그러지게)
    ax.set_title("Ant paths through maze (random actions)\n[wall coords hardcoded]")
    ax.legend()
    plt.savefig("outputs/images/test_env_path.png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    print("✅ 경로 그림 저장: outputs/images/test_env_path.png")

    # ── 4) 영상용 action 시퀀스 1개 확보 (seed 고정) ───
    VIDEO_SEED = 123
    _, _, actions = run_episode(env, seed=VIDEO_SEED)
    env.close()

    # ── 5) 같은 action을 두 카메라로 리플레이 (camera_id=-1 → 자유 카메라) ──
    env_top = gym.make("AntMaze-v0", render_mode="rgb_array",
                       camera_id=-1, default_camera_config=TOPDOWN_CAM)
    frames_top = replay_actions(env_top, actions, seed=VIDEO_SEED)
    env_top.close()

    env_follow = gym.make("AntMaze-v0", render_mode="rgb_array",
                          camera_id=-1, default_camera_config=FOLLOW_CAM)
    frames_follow = replay_actions(env_follow, actions, seed=VIDEO_SEED, follow=True)
    env_follow.close()

    # ── 6) 영상 2개 저장 (macro_block_size=None → ffmpeg 자동 리사이즈 방지) ──
    os.makedirs("outputs/videos/00_셋업_환경·전문가", exist_ok=True)
    imageio.mimsave("outputs/videos/00_셋업_환경·전문가/환경테스트_탑다운.mp4",
                    frames_top, fps=30, macro_block_size=None)
    imageio.mimsave("outputs/videos/00_셋업_환경·전문가/환경테스트_추적카메라.mp4",
                    frames_follow, fps=30, macro_block_size=None)
    print(f"✅ 영상 저장: outputs/videos/00_셋업_환경·전문가/환경테스트_탑다운.mp4 ({len(frames_top)} 프레임)")
    print(f"✅ 영상 저장: outputs/videos/00_셋업_환경·전문가/환경테스트_추적카메라.mp4  ({len(frames_follow)} 프레임)")

    print("\n🎉 1.3 완료 — 환경 동작 확인 끝!")


if __name__ == "__main__":
    main()

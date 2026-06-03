"""
설치 확인 스크립트.
MuJoCo + Gymnasium이 제대로 깔렸는지 Ant-v5 환경으로 검증한다.
- 환경 로드(위치 포함) → 10스텝 무작위 실행 → 관측값 출력 → 첫 프레임 저장
"""
import os
# macOS 오프스크린 렌더 안정화. 검은 화면/에러 나면 "egl" 또는 "osmesa"로 바꿔 시도.
os.environ.setdefault("MUJOCO_GL", "glfw")

import gymnasium as gym
import matplotlib.pyplot as plt


def main():
    # exclude_current_positions_from_observation=False:
    #   기본값(True)이면 obs에서 x,y가 빠진다 → 미로 내비게이션엔 위치가 필수라 False로 둔다.
    env = gym.make(
        "Ant-v5",
        render_mode="rgb_array",
        exclude_current_positions_from_observation=False,
    )

    obs, info = env.reset(seed=42)
    env.action_space.seed(42)  # 무작위 행동 재현성

    print(f"관측 공간(observation space): {env.observation_space}")
    print(f"행동 공간(action space)     : {env.action_space}")
    print(f"초기 관측 shape            : {obs.shape}")
    print(f"ant (x, y)                 : {obs[:2]}  ← 위치가 obs에 들어왔는지 확인\n")

    first_frame = None
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step + 1:2d} | reward={reward:7.3f} | obs[:3]={obs[:3]}")
        if step == 0:
            first_frame = env.render()

    if first_frame is not None:
        os.makedirs("outputs/images", exist_ok=True)  # 폴더 없으면 생성
        plt.imshow(first_frame)
        plt.axis("off")
        plt.title("Ant-v5 first frame")
        plt.savefig("outputs/images/check_install_frame.png", bbox_inches="tight", dpi=120)
        print("\n✅ 프레임 저장 완료: outputs/images/check_install_frame.png")

    env.close()
    print("✅ 설치 확인 완료 — 환경이 정상 작동합니다.")


if __name__ == "__main__":
    main()

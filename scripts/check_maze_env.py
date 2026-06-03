import os
os.environ.setdefault("MUJOCO_GL", "glfw")  # 검은화면이면 egl/osmesa
import gymnasium as gym
import matplotlib.pyplot as plt
import src.envs  # ← register 실행 (필수)

env = gym.make("AntMaze-v0", render_mode="rgb_array")
obs, info = env.reset(seed=0)
print("observation_space:", env.observation_space)        # (109,) 기대
print("reset obs shape  :", obs.shape)                     # (109,) — step과 동일해야 함
print("obs 끝 2개(목표상대벡터):", obs[-2:])              # 시작(0,0)→목표(0,6) 이므로 ≈ [0, 6]

for i in range(5):
    a = env.action_space.sample()
    obs, r, term, trunc, info = env.step(a)
    assert obs.shape == env.observation_space.shape, (obs.shape, env.observation_space.shape)
    print(f"step{i}: r={r:7.3f} dist={info['dist_to_goal']:.2f} term={term}")

frame = env.render()
os.makedirs("outputs/images", exist_ok=True)
plt.imshow(frame); plt.axis("off"); plt.title("AntMaze-v0")
plt.savefig("outputs/images/maze_env_frame.png", bbox_inches="tight", dpi=120)
env.close()
print("✅ 통과: reset/step obs 차원 일치, 미로 렌더 저장 (outputs/images/maze_env_frame.png)")
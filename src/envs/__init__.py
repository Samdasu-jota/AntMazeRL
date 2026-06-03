"""환경 패키지 초기화 + Gymnasium에 AntMazeEnv 등록."""
from gymnasium.envs.registration import register
from src.envs.ant_maze_env import AntMazeEnv  # noqa: F401

register(
    id="AntMaze-v0",
    entry_point="src.envs.ant_maze_env:AntMazeEnv",
    max_episode_steps=1000,   # 시간초과 = truncated (TimeLimit 래퍼)
)

# Stage 0 — 빈 평지(가운데 벽 제거) + 'direct' 보상(진행/stall/시간/축소된 도달보너스).
#   목적: "개미가 목표까지 제대로 걷는가?"를 미로 없이 먼저 검증.
#   per-run 으로 env_kwargs(예: {goal_y: 4.0})를 gym.make에 넘기면 아래 기본값을 덮어쓴다.
register(
    id="AntMazeOpen-v0",
    entry_point="src.envs.ant_maze_env:AntMazeEnv",
    max_episode_steps=1000,
    kwargs=dict(obstacle=False, reward_mode="direct", goal_y=3.0,
                goal_bonus=50.0, progress_coef=10.0, energy_coef=0.1,
                stall_penalty=1.0, time_penalty=0.05,
                stall_speed=0.1, stall_dist=1.0),
)

# Stage 1 — 기둥 복귀(obstacle=True) + 'direct' 보상을 그대로 쓰되, WaypointFollower 래퍼가
#   goal_pos를 현재 서브목표로 바꿔치기해 웨이포인트를 따라가게 한다.
#   ⚠️ goal_bonus=0: env가 서브목표마다 +50을 흘리는 것을 막고 보너스는 래퍼가 전담.
register(
    id="AntMazeWaypoint-v0",
    entry_point="src.envs.ant_maze_env:AntMazeEnv",
    max_episode_steps=1000,
    kwargs=dict(obstacle=True, reward_mode="direct", goal_y=6.0,
                goal_bonus=0.0, progress_coef=10.0, energy_coef=0.1,
                stall_penalty=1.0, time_penalty=0.05,
                stall_speed=0.1, stall_dist=1.0),
)
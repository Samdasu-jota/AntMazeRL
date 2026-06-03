"""
Stage 1 래퍼 — 검증된 보행 정책 위에 '고전 길찾기(A*/웨이포인트)'를 얹는다.

핵심 아이디어(업계 표준 분리, ANYmal/Barkour/Skill-Nav):
  - 길찾기 = 고전(A* 또는 하드코딩 웨이포인트)이 '서브목표' 시퀀스를 준다.
  - 보행 = RL 정책이 '현재 서브목표'를 향해 걷는다.
이 래퍼는 AntMazeEnv를 **전혀 수정하지 않고**(`ant_maze_env.py` 무편집) env의
`goal_pos`를 매 스텝 현재 서브목표로 바꿔치기만 한다. 그러면 env의 자체 `_get_obs`
(목표 상대벡터)와 `_step_direct`(목표까지 진행 보상)가 그대로 서브목표 기준으로 동작한다.

왜 별도 래퍼인가:
  - Stage 0(AntMazeOpen-v0)·미로(AntMaze-v0) env 동작을 byte-단위로 보존(재현성).
  - 검증된 'direct' 보상(_step_direct)만 재사용 — 실패한 미로 'waypoint' 보상 분기(실험#7=0%)는 안 씀.
"""

import numpy as np
import gymnasium as gym

from src.envs.ant_maze_env import WAYPOINTS, WP_REACH, GOAL_RADIUS


class WaypointFollower(gym.Wrapper):
    """현재 서브목표를 향해 obs/보상을 주고, 도달하면 다음 서브목표로 넘어간다.

    매 스텝 env.unwrapped.goal_pos를 현재 서브목표로 세팅 → env의 _get_obs가
    '서브목표 상대벡터'를, _step_direct가 '서브목표까지 진행 보상'을 자동 계산.
    종료/성공 판정은 **최종 목표** 기준으로 래퍼가 덮어쓴다(서브목표에선 종료 안 함).

    ⚠️ AntMazeWaypoint-v0는 goal_bonus=0으로 등록 → env가 서브목표마다 보너스를
       흘리는 것을 막고, 보너스는 래퍼가 전담(서브목표 advance=+subgoal_bonus,
       최종 도달=+final_bonus).
    """

    def __init__(self, env, waypoints=None, wp_reach=WP_REACH, final_goal=None,
                 goal_radius=GOAL_RADIUS, subgoal_bonus=10.0, final_bonus=50.0):
        super().__init__(env)
        wps = waypoints if waypoints is not None else WAYPOINTS
        self.waypoints = [np.asarray(w, dtype=float) for w in wps]
        self.wp_reach = wp_reach
        self.final_goal = np.asarray(
            final_goal if final_goal is not None else self.waypoints[-1], dtype=float)
        self.goal_radius = goal_radius
        self.subgoal_bonus = subgoal_bonus
        self.final_bonus = final_bonus
        self._wp_idx = 0

    def reset(self, **kwargs):
        self._wp_idx = 0
        u = self.env.unwrapped
        u.goal_pos = self.waypoints[0].copy()   # reset의 _get_obs가 서브목표0을 보게
        u._prev_goal_dist = None
        obs, info = self.env.reset(**kwargs)
        info["wp_idx"] = self._wp_idx
        return obs, info

    def step(self, action):
        u = self.env.unwrapped
        ant_xy = u.data.qpos[:2].copy()          # 스텝 '전' 위치로 advance 판정

        advanced = False
        if (self._wp_idx < len(self.waypoints) - 1 and
                np.linalg.norm(self.waypoints[self._wp_idx] - ant_xy) < self.wp_reach):
            self._wp_idx += 1
            u._prev_goal_dist = None             # ★ 목표 전환 시 진행보상 스파이크 제거
            advanced = True
        u.goal_pos = self.waypoints[self._wp_idx].copy()

        obs, reward, terminated, truncated, info = self.env.step(action)

        # 종료/성공은 '최종 목표' 기준으로 덮어쓴다(서브목표 도달로는 종료 안 함)
        post_xy = u.data.qpos[:2]
        final_dist = float(np.linalg.norm(self.final_goal - post_xy))
        reached_final = final_dist < self.goal_radius
        fell = bool(u.data.qpos[2] < 0.2 or u._flipped())   # ★ 키 OR 전복(자세) 종료

        if advanced:
            reward += self.subgoal_bonus
        if reached_final:
            reward += self.final_bonus

        terminated = bool(reached_final or fell)
        info["reached_goal"] = reached_final     # ★ final_eval이 이걸 읽음(최종 목표)
        info["wp_idx"] = self._wp_idx
        info["dist_to_final"] = final_dist
        return obs, reward, terminated, truncated, info


class RandomGoalDirection(gym.Wrapper):
    """매 reset마다 목표를 임의 방향·거리로 둔다 → 보행 정책이 '어느 방향이든 조향'하게 학습.

    Stage 0 워커는 +y 목표로만 학습돼 목표벡터를 무시(방향맹). 이 래퍼로 방향을
    랜덤화해 조향 인센티브를 직접 준다(legged_gym의 randomized-command 레시피).

    목표는 경계벽 안쪽 reachable box 안으로 클램프(ray-box) → 도달 불가능 목표로
    성공률 상한이 깎이는 것 방지. 시각 마커(고정)는 안 맞지만 cosmetic(학습은 렌더 안 함).
    """

    def __init__(self, env, dist_min=2.0, dist_max=6.5,
                 box=(-3.4, 3.4, -1.4, 7.4), seed=None):
        super().__init__(env)
        self.dist_min = dist_min
        self.dist_max = dist_max
        self.box = box                            # (xmin, xmax, ymin, ymax) 경계벽 안쪽
        self.rng = np.random.default_rng(seed)

    def _max_reach(self, ang):
        """원점에서 방향 ang을 따라 box 경계까지의 거리."""
        c, s = np.cos(ang), np.sin(ang)
        xmin, xmax, ymin, ymax = self.box
        ts = []
        if c > 1e-6:
            ts.append(xmax / c)
        elif c < -1e-6:
            ts.append(xmin / c)
        if s > 1e-6:
            ts.append(ymax / s)
        elif s < -1e-6:
            ts.append(ymin / s)
        return min(ts) if ts else self.dist_max

    def reset(self, **kwargs):
        ang = float(self.rng.uniform(0, 2 * np.pi))
        hi = min(self.dist_max, self._max_reach(ang))
        lo = min(self.dist_min, 0.7 * hi)
        d = float(self.rng.uniform(lo, hi))
        u = self.env.unwrapped
        u.goal_pos = np.array([d * np.cos(ang), d * np.sin(ang)], dtype=float)
        u._prev_goal_dist = None
        return self.env.reset(**kwargs)


class RandomWaypointSequence(gym.Wrapper):
    """빈 평지에서 '목표가 에피소드 중간에 바뀌는' 학습 → **달리는 중 방향 전환** 습득.

    Stage 0/전방향 학습은 목표가 에피소드 내내 '고정'이라, 개미는 '출발 자세에서
    한 번 꺾기'만 배웠다. 미로는 다리를 끝낼 때마다 *움직이던 중* 새 방향으로 다시
    꺾어야 한다. 이 래퍼는 목표 도달 즉시 새 랜덤 방향 목표로 교체(최대 max_goals개)해
    그 mid-episode 조향을 직접·자주 연습시킨다(legged_gym command-resampling 레시피).

    종료 = 넘어짐(z<0.2) 또는 max_goals 도달. 도달마다 env의 goal_bonus(+50)가 보상.
    """

    def __init__(self, env, max_goals=5, dist_range=(2.0, 4.0),
                 box=(-3.4, 3.4, -1.4, 7.4), reach_clear=1.5,
                 max_turn_angle=None, seed=None):
        super().__init__(env)
        self.max_goals = max_goals
        self.dist_range = dist_range
        self.box = box
        self.reach_clear = reach_clear
        # 각도 커리큘럼: 새 목표를 '현재 진행방향 ±max_turn_angle' 안으로 바이어스.
        # None = 기존 동작(전방향 균등). 좁게 시작(±30°)→넓게(±90°) 단계 학습.
        self.max_turn_angle = max_turn_angle
        self.rng = np.random.default_rng(seed)
        self._n = 0
        self._last_heading = 0.0               # 거의 정지 시 heading fallback(직전 목표 방향)

    def _current_heading(self, u, fallback):
        """world-frame 진행방향(수평속도 qvel[:2]). 거의 정지면 fallback(직전 목표 방향).
        정지 시 qvel 방향은 노이즈라 그대로 쓰면 각도 바이어스가 무력화됨 → fallback."""
        v = u.data.qvel[:2]
        if float(np.linalg.norm(v)) < 0.15:
            return fallback
        return float(np.arctan2(v[1], v[0]))

    def _sample_goal(self, from_xy, heading=None):
        xmin, xmax, ymin, ymax = self.box
        for _ in range(20):
            if self.max_turn_angle is None or heading is None:
                ang = self.rng.uniform(0, 2 * np.pi)            # 전방향(기존) 또는 첫 목표
            else:
                ang = heading + self.rng.uniform(-self.max_turn_angle, self.max_turn_angle)
            d = self.rng.uniform(*self.dist_range)
            g = np.array([from_xy[0] + d * np.cos(ang),
                          from_xy[1] + d * np.sin(ang)], dtype=float)
            g[0] = float(np.clip(g[0], xmin, xmax))
            g[1] = float(np.clip(g[1], ymin, ymax))
            if np.linalg.norm(g - from_xy) >= self.reach_clear:
                self._last_heading = ang        # 다음 fallback용으로 의도한 방향 기억
                return g
        return np.array([float(np.clip(from_xy[0] + 2.0, xmin, xmax)),
                         float(np.clip(from_xy[1], ymin, ymax))], dtype=float)

    def reset(self, **kwargs):
        self._n = 0
        u = self.env.unwrapped
        # 첫 목표는 출발 자세(정지)에서 → heading 없음(균등). _sample_goal이 _last_heading 설정.
        u.goal_pos = self._sample_goal(np.array([0.0, 0.0]))
        u._prev_goal_dist = None
        obs, info = self.env.reset(**kwargs)
        info["n_reached"] = 0
        return obs, info

    def step(self, action):
        u = self.env.unwrapped
        obs, reward, _, trunc, info = self.env.step(action)
        reached = bool(info.get("reached_goal", False))
        fell = bool(u.data.qpos[2] < 0.2 or u._flipped())   # ★ 키 OR 전복(자세) 종료
        if reached:
            self._n += 1
            if self._n < self.max_goals and not fell:
                # 움직이던 중 새 목표 → 현재 진행방향 기준 ±max_turn_angle로 바이어스
                heading = self._current_heading(u, self._last_heading)
                u.goal_pos = self._sample_goal(u.data.qpos[:2].copy(), heading=heading)
                u._prev_goal_dist = None
                obs = u._get_obs()              # 새 목표 기준으로 obs 갱신(1스텝 지연 제거)
        terminated = bool(fell or self._n >= self.max_goals)
        info["n_reached"] = self._n
        return obs, reward, terminated, trunc, info

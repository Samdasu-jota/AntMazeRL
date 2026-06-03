"""
AntMazeEnv — MuJoCo Ant-v5를 확장한 커스텀 미로 환경.

기본 Ant는 "앞으로 빨리 달리기"가 목표지만,
이 환경은 "미로를 통과해 목표점에 도달하기"가 목표다.

핵심 변경점:
1. MJCF XML로 미로 벽 추가
2. 시작점/목표점 정의
3. 보상 = 속도(목표 접근) + 에너지 효율 + 벽 회피(단순화 버전)
4. 종료 = 목표 도달 or 넘어짐 (시간초과는 TimeLimit 래퍼가 처리)
5. 목표 위치에 빨간 공(시각 표시)
6. obs에 "목표까지의 상대벡터" 추가 (길 찾기에 필수)
"""

import os
import tempfile
import numpy as np
from gymnasium.envs.mujoco.ant_v5 import AntEnv
from gymnasium.spaces import Box


# ── 미로 설정 (MuJoCo 단위 = 미터) ──────────────────────
START_POS = np.array([0.0, 0.0])      # 개미 시작 (x, y)
GOAL_POS = np.array([0.0, 6.0])       # 목표 위치 (x, y)
GOAL_RADIUS = 1.0                     # 목표 도달 판정 반경(m)

# ── 자세(전복) 감지 ─────────────────────────────────────
#   up_z = 몸통 up축의 world z성분 (+1 똑바로 · 0 옆으로 · -1 완전 전복).
#   기존 종료는 z<0.2(키)만 봐서 '뒤집힌 채 기어가기'(z>0.2 유지)를 못 잡았다.
#   진단(eval_flip): 평지 워커 1/3, 미로 워커 ~전부가 (거의)완전 전복인데 fell 0.
#   UP_THRESH=0.0 → 옆으로 누움 이상이면 종료. 평지 min up_z 히스토그램이 이봉형이라
#   [0.0,0.5)가 비어 정상보행(≥0.5)은 안 건드리고 깊은 전복(≤-0.5)만 잡는 안전 경계.
UP_THRESH_DEFAULT = 0.0


def compute_up_z(quat):
    """쿼터니언(qw,qx,qy,qz)에서 몸통 up축의 world z성분. up_z = 1 - 2*(qx²+qy²).
    qpos[3:7] 또는 obs[3:7] 어느 쪽이든 동일(레이아웃 같음)."""
    qx, qy = quat[1], quat[2]
    return 1.0 - 2.0 * (qx * qx + qy * qy)

# ── 웨이포인트 경로 (실험#7): 가운데 벽을 오른쪽으로 우회 → 위 → 목표 ──
#   ⚠️ scripted_expert.py가 이 파일의 GOAL_POS를 import → 거꾸로 import하면 순환참조.
#      그래서 값만 복제(전문가 WAYPOINTS/WP_REACH와 동일 유지).
WAYPOINTS = [
    np.array([2.5, 0.0]),
    np.array([2.5, 6.3]),
    GOAL_POS,
]
WP_REACH = 0.8                        # 이 거리 안에 들면 다음 웨이포인트로
# 웨이포인트 간 leg 길이(누적 '남은 경로거리' 계산용 — 매 스텝 재계산 방지)
_WP_LEG = [float(np.linalg.norm(WAYPOINTS[i + 1] - WAYPOINTS[i]))
           for i in range(len(WAYPOINTS) - 1)]


def make_maze_xml(base_xml_path: str, obstacle: bool = True,
                  goal_y: float = 6.0, pillar_half_len: float = 3.0) -> str:
    """기본 Ant XML에 미로 벽 + 목표 공을 끼워넣은 임시 XML을 만들어 경로 반환.

    obstacle=False 면 가운데 칸막이(wall_mid)를 빼서 '빈 평지(open plane)'로 만든다
    (Stage 0). 바깥 경계 4개는 유지 → 개미가 무한히 벗어나지 않게 한다.
    goal_y 로 목표 공 위치를 옮긴다(커리큘럼 3m→6m).
    """
    with open(base_xml_path, "r") as f:
        xml = f.read()

    # geom type=box: pos="중심 x y z", size="각 축 절반길이 hx hy hz"
    #   ⚠️ obstacle=True 경로는 기존 미로 실험(#1~#7) 재현을 위해 문자열을 그대로 유지.
    # pillar_half_len: 기둥 길이의 절반. 기본 3.0 → y∈[0,6](6m). 중앙 (0,3) 고정.
    #   1.0이면 1/3 축소(y∈[2,4]) → 스폰(0,0)·목표(0,6)가 기둥과 분리.
    #   :g 포맷으로 3.0→"3" 렌더 → 기본값일 때 기존 geom 문자열 그대로(회귀 안전).
    wall_mid = (f"""
        <!-- 가운데 칸막이: 직진을 막아 U자 경로 강제 (중앙(0,3), 길이 {2 * pillar_half_len:g}m) -->
        <geom name="wall_mid" type="box" pos="0 3 0.5" size="0.3 {pillar_half_len:g} 0.5"
              rgba="0.5 0.5 0.5 1" contype="1" conaffinity="1"/>"""
                if obstacle else "")
    maze_geoms = wall_mid + f"""
        <!-- 바깥 경계 4개 -->
        <geom name="wall_left"  type="box" pos="-4 3 0.5" size="0.3 5 0.5"
              rgba="0.5 0.5 0.5 1" contype="1" conaffinity="1"/>
        <geom name="wall_right" type="box" pos="4 3 0.5"  size="0.3 5 0.5"
              rgba="0.5 0.5 0.5 1" contype="1" conaffinity="1"/>
        <geom name="wall_top"   type="box" pos="0 8 0.5"  size="4 0.3 0.5"
              rgba="0.5 0.5 0.5 1" contype="1" conaffinity="1"/>
        <geom name="wall_bot"   type="box" pos="0 -2 0.5" size="4 0.3 0.5"
              rgba="0.5 0.5 0.5 1" contype="1" conaffinity="1"/>
        <!-- 목표 표시용 빨간 공 (충돌 없음) -->
        <geom name="goal_marker" type="sphere" pos="0 {goal_y} 0.5" size="0.5"
              rgba="1 0 0 0.6" contype="0" conaffinity="0"/>
    """
    xml = xml.replace("</worldbody>", maze_geoms + "\n</worldbody>")

    tmp = tempfile.NamedTemporaryFile(suffix=".xml", delete=False, mode="w")
    tmp.write(xml)
    tmp.close()
    return tmp.name


class AntMazeEnv(AntEnv):
    """Ant-v5를 상속받아 미로 내비게이션 환경으로 변형."""

    def __init__(self, obstacle=True, goal_y=6.0, reward_mode="waypoint",
                 goal_bonus=200.0, progress_coef=10.0, energy_coef=0.1,
                 stall_penalty=0.0, time_penalty=0.0,
                 stall_speed=0.1, stall_dist=1.0, start_xy=(0.0, 0.0),
                 pillar_half_len=3.0,
                 up_thresh=UP_THRESH_DEFAULT, up_bonus_coef=0.0,
                 include_up_z=False, **kwargs):
        # ⚠️ Stage-0 파라미터는 '명시 인자'로 받아 self에 저장한다. **kwargs로 흘리면
        #    부모 AntEnv가 모르는 인자라며 TypeError를 낸다(빈 평지 env가 즉시 크래시).
        self.reward_mode = reward_mode
        self.goal_pos = np.array([0.0, goal_y], dtype=float)
        self.goal_bonus = goal_bonus
        self.progress_coef = progress_coef
        self.energy_coef = energy_coef
        self.stall_penalty = stall_penalty
        self.time_penalty = time_penalty
        self.stall_speed = stall_speed
        self.stall_dist = stall_dist
        self.start_xy = tuple(start_xy)     # 스폰 오프셋(기둥 밑동(0,0) 끼임 회피). (0,0)=기존 동작.
        # 자세(전복) 제어 — 모두 기본값은 무동작에 가깝게(회귀 안전).
        #   up_thresh: up_z가 이 밑이면 종료(전복). 0.0=옆으로 누움 이상.
        #   up_bonus_coef: 매 스텝 +up_bonus_coef·max(0,up_z) 직립 보너스(+부호=suicide 안전). 0=꺼짐.
        #   include_up_z: obs 끝에 up_z 1차원 추가(→ 정규화통계/네트워크 입력 110차원, scratch 전용).
        #   ⚠️ _get_obs가 super().__init__ 중 호출되므로 include_up_z는 그 '전'에 설정돼야 함.
        self.up_thresh = up_thresh
        self.up_bonus_coef = up_bonus_coef
        self.include_up_z = include_up_z

        # 1) 기본 Ant XML 경로 (설치된 gymnasium 패키지 내부)
        from gymnasium.envs.mujoco import ant_v5
        base_dir = os.path.dirname(ant_v5.__file__)
        base_xml = os.path.join(base_dir, "assets", "ant.xml")

        # 2) 미로 벽 주입한 새 XML 생성 (obstacle=False면 가운데 벽 제거 = 빈 평지)
        self.maze_xml_path = make_maze_xml(base_xml, obstacle=obstacle, goal_y=goal_y,
                                           pillar_half_len=pillar_half_len)
        self.obstacle = obstacle

        # 3) 부모 초기화. exclude_...=False → obs에 x,y 포함(필수)
        super().__init__(
            xml_file=self.maze_xml_path,
            exclude_current_positions_from_observation=False,
            **kwargs,
        )

        # 4) obs 끝에 [목표 상대벡터 2차원] (+ include_up_z면 up_z 1차원) 추가
        #    (부모가 설정한 실제 크기에 더하므로 버전 차이에 안전)
        extra = 2 + (1 if self.include_up_z else 0)
        obs_dim = self.observation_space.shape[0] + extra
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )

        # 5) 보상 진행 추적 상태
        #    waypoint 모드(미로): 웨이포인트 idx + 직전 '남은 경로거리'
        #    direct  모드(평지): 직전 '목표까지 직선거리'(_prev_goal_dist, 별도 변수로 분리)
        self._wp_idx = 0
        self._prev_path_dist = None
        self._prev_goal_dist = None

    def _get_obs(self):
        """기본 관측 뒤에 [목표까지 상대벡터 x,y] (+ include_up_z면 up_z)를 붙인다."""
        base_obs = super()._get_obs()
        ant_xy = self.data.qpos[:2].copy()
        rel_to_goal = self.goal_pos - ant_xy
        if self.include_up_z:
            up_z = compute_up_z(self.data.qpos[3:7])
            return np.concatenate([base_obs, rel_to_goal, [up_z]])
        return np.concatenate([base_obs, rel_to_goal])

    def _up_z(self):
        """현재 몸통 up_z(라이브 qpos). env·래퍼(u._up_z())의 전복 판정 단일 출처."""
        return float(compute_up_z(self.data.qpos[3:7]))

    def _flipped(self):
        """몸통이 임계치 너머로 기울었는가(전복). 종료 판정용."""
        return bool(self._up_z() < self.up_thresh)

    def step(self, action):
        # 부모 step으로 물리 한 칸 진행 (부모 reward는 무시, truncated는 래퍼가 채움)
        _, _, _, truncated, info = super().step(action)

        ant_xy = self.data.qpos[:2].copy()
        dist_to_goal = np.linalg.norm(self.goal_pos - ant_xy)
        z = self.data.qpos[2]
        reached_goal = dist_to_goal < GOAL_RADIUS

        if self.reward_mode == "direct":
            return self._step_direct(action, dist_to_goal, z, reached_goal, truncated, info)

        # ── waypoint 모드(미로, 실험#7): 아래는 기존 코드 그대로 유지 ──────────────
        # (a) 속도 보상: 유클리드 직선거리 대신 '웨이포인트 경로 남은거리' 감소로 보상.
        #     → 미로가 요구하는 우회((2.5,0)行)를 처벌하지 않고 경로를 따라 길을 안내한다.
        if (self._wp_idx < len(WAYPOINTS) - 1 and
                np.linalg.norm(WAYPOINTS[self._wp_idx] - ant_xy) < WP_REACH):
            self._wp_idx += 1                        # 현재 웨이포인트 도달 → 다음 목표로
        path_dist = (np.linalg.norm(WAYPOINTS[self._wp_idx] - ant_xy)
                     + sum(_WP_LEG[self._wp_idx:]))   # 현재 wp까지 거리 + 남은 leg 합
        if self._prev_path_dist is None:
            self._prev_path_dist = path_dist
        progress = self._prev_path_dist - path_dist
        speed_reward = 10.0 * progress
        self._prev_path_dist = path_dist

        # (b) 에너지 페널티: 관절 힘 제곱합
        energy_penalty = 0.1 * np.sum(np.square(action))

        # (c) (단순화) 넘어짐 페널티: 몸통 z가 낮으면 -  ※3.2에서 진짜 벽충돌로 교체
        collision_penalty = 1.0 if z < 0.3 else 0.0

        # (d) 목표 도달 보너스 (실험#6): 지금까진 도달이 return에 안 보였음 → 강한 완수 인센티브
        goal_bonus = 200.0 if reached_goal else 0.0

        # (e) 직립 보너스(+부호=suicide 안전): 똑바로 설수록 +. up_bonus_coef=0이면 무동작.
        up_z = self._up_z()
        up_bonus = self.up_bonus_coef * max(0.0, up_z)

        reward = speed_reward - energy_penalty - collision_penalty + goal_bonus + up_bonus

        fell_over = z < 0.2 or self._flipped()      # ★ 키 OR 전복(자세) 종료
        terminated = bool(reached_goal or fell_over)

        info.update({
            "speed_reward": speed_reward,
            "energy_penalty": energy_penalty,
            "collision_penalty": collision_penalty,
            "goal_bonus": goal_bonus,
            "up_bonus": up_bonus,
            "up_z": up_z,
            "dist_to_goal": dist_to_goal,
            "reached_goal": reached_goal,
        })
        return self._get_obs(), reward, terminated, truncated, info

    def _step_direct(self, action, dist_to_goal, z, reached_goal, truncated, info):
        """Stage 0 'direct' 보상: 빈 평지에서 목표까지 '제대로 걷기'를 직격.

        리서치(ETH ANYmal / Ng 1999 / Rudin 2022) 반영:
          - 진행 보상 = potential 기반(목표까지 직선거리 감소). 배회 루프는 net≈0.
          - stall 페널티 = 느린데 + 멀면 -1 → '종일 버티기(wander-out-the-clock)' 처벌.
          - 시간 페널티 = 매 스텝 약간 - → 빨리 끝내는 게 이득.
          - 도달 보너스 = +200이 아니라 +50(어차피 큰 보너스는 발화 안 했음).
        """
        if self._prev_goal_dist is None:
            self._prev_goal_dist = dist_to_goal
        progress = self._prev_goal_dist - dist_to_goal
        self._prev_goal_dist = dist_to_goal

        speed = float(np.linalg.norm(self.data.qvel[:2]))   # world-frame 수평 속도
        speed_reward = self.progress_coef * progress
        energy_penalty = self.energy_coef * np.sum(np.square(action))
        collision_penalty = 1.0 if z < 0.3 else 0.0         # 배깔기/주저앉음 억제
        stall_pen = (self.stall_penalty
                     if (speed < self.stall_speed and dist_to_goal > self.stall_dist)
                     else 0.0)
        goal_bonus = self.goal_bonus if reached_goal else 0.0
        # 직립 보너스(+부호=suicide 안전): 똑바로 설수록 +. up_bonus_coef=0이면 무동작.
        up_z = self._up_z()
        up_bonus = self.up_bonus_coef * max(0.0, up_z)

        reward = (speed_reward + goal_bonus + up_bonus
                  - energy_penalty - collision_penalty
                  - stall_pen - self.time_penalty)

        fell_over = z < 0.2 or self._flipped()      # ★ 키 OR 전복(자세) 종료
        terminated = bool(reached_goal or fell_over)

        # ⚠️ 기존 info 키(speed_reward/energy_penalty/collision_penalty)를 그대로 채워
        #    ProgressCallback이 조용히 0만 찍는 일을 막는다. + Stage-0 신규 항 추가.
        info.update({
            "speed_reward": speed_reward,
            "energy_penalty": energy_penalty,
            "collision_penalty": collision_penalty,
            "goal_bonus": goal_bonus,
            "up_bonus": up_bonus,
            "up_z": up_z,
            "stall_penalty": stall_pen,
            "time_penalty": self.time_penalty,
            "progress": progress,
            "speed": speed,
            "dist_to_goal": dist_to_goal,
            "reached_goal": reached_goal,
        })
        return self._get_obs(), reward, terminated, truncated, info

    def reset_model(self):
        """에피소드 시작. 웨이포인트/경로거리 추적 초기화 — obs는 부모가 (오버라이드된)
        _get_obs로 이미 상대벡터까지 붙여 반환하므로 그대로 돌려준다."""
        self._wp_idx = 0
        self._prev_path_dist = None
        self._prev_goal_dist = None
        obs = super().reset_model()       # ★ 이중추가 버그 수정: 여기서 또 붙이지 않음
        sx, sy = self.start_xy
        if sx != 0.0 or sy != 0.0:        # 스폰 오프셋: 기둥 밑동(0,0) 끼임 회피 (기본 (0,0)=무변경)
            qpos = self.data.qpos.copy()
            qvel = self.data.qvel.copy()
            qpos[0] += sx
            qpos[1] += sy
            self.set_state(qpos, qvel)
            obs = self._get_obs()
        return obs

    def close(self):
        """env 종료 시 생성했던 임시 XML 정리(임시파일 누수 방지)."""
        super().close()
        try:
            os.remove(self.maze_xml_path)
        except (OSError, AttributeError):
            pass
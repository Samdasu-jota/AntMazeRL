"""
scripted_expert.py — 규칙 기반(학습 없음) 전문가 정책.

핵심 아이디어:
  4족 보행은 "한 방향으로 힘 주기"로는 안 걷는다. 다리를 주기적으로(sin) 흔드는
  '걸음새(gait)'가 필요하다. 여기서는 대각선 트로트(diagonal trot) 걸음새를 직접
  설계하고, 몸통 방향(yaw)을 읽어 목표 쪽으로 조향하며, 벽을 피해 웨이포인트를 따라간다.

  - 학습 전혀 없음(if/else + sin). 신경망 아님.
  - 목적: 완벽한 보행이 아니라 "대부분 목표 쪽으로 전진하는" 적당한 시범 데이터
          생성 (2.2 behavior cloning의 학습 재료).

⚠️ 토크 제어(actuator=motor)라 sin 걸음새의 정확한 부호/진폭은 미리 알 수 없다.
   scripts/preview_expert.py 로 A단계(직진)부터 눈으로 보며 아래 상수를 튜닝한다.

obs 구조 (1.2/검토에서 확정):
  obs[0:2]  = 몸통 (x, y)         (world)
  obs[2]    = 몸통 z (높이)
  obs[3:7]  = 몸통 자세 쿼터니언 (qw, qx, qy, qz)
  obs[7:15] = 8개 관절각  ...
  obs[-2:]  = 목표 상대벡터 (GOAL_POS - ant_xy, world frame)

Ant-v5 actuator 순서 (gymnasium ant.xml 실측):
  idx 0,1 = hip_4, ankle_4  (right-back leg,  몸통기준 +x,-y)
  idx 2,3 = hip_1, ankle_1  (front-left leg,  +x,+y)
  idx 4,5 = hip_2, ankle_2  (front-right leg, -x,+y)
  idx 6,7 = hip_3, ankle_3  (back-left leg,   -x,-y)
모든 hip = 수직(yaw) 경첩(±30°). ankle 범위 부호가 다리마다 다름
(leg1,4 = +30..70 / leg2,3 = -70..-30) → 아래 ank_sign 으로 반영.
"""

import numpy as np

from src.envs.ant_maze_env import GOAL_POS


# ── 물리 스텝 간격(초). Ant-v5 기본 frame_skip(5) × timestep(0.01) = 0.05 ──
DT = 0.05

# ── 걸음새/조향 상수 (헤드리스 파라미터 탐색으로 튜닝한 값) ──────────
#   핵심 교훈:
#   1) ankle에 '펴짐 바이어스(ANK_BIAS)'를 줘야 몸통이 안 무너지고 선다(z≈0.67).
#      바이어스 없이 sin만 주면 다리가 반주기 동안 힘이 0 → 배 깔고 주저앉음.
#   2) hip 부호(HIP_SIGN)를 다리마다 다르게 줘야 몸통이 안 돈다. 전부 같은 부호면
#      yaw 토크가 누적돼 제자리 회전(스핀)한다.
#   3) 조향 = 모든 hip에 공통 오프셋(turn)을 더해 몸통을 돌린다(실측: 권한 충분).
FREQ = 1.2          # 걸음 주파수 (cycles/sec)
HIP_AMP = 0.6       # hip(다리 앞뒤 스윙) 토크 진폭 → 추진
ANK_BIAS = 0.8      # ankle 펴짐 바이어스 → 몸통을 세워 지지 (가장 중요)
ANK_AMP = 0.3       # ankle 진동 진폭 (스윙 때 들고 stance 때 디딤)
ANK_LAG = np.pi / 2 # ankle 위상 지연

# 시작 워밍업: 처음 몇 스텝 동안 진동/조향 진폭을 0→1로 서서히 키운다.
#   풀진폭 토크를 서있는 개미에 즉시 때리면 휘청→전복. 램프로 초반 전복을 줄임.
#   (ank_bias는 즉시 적용 — 몸통을 받쳐야 하므로.)
WARMUP_STEPS = 40

# 전복 판정(데이터 수집 시 사용): 몸통 z가 이 아래면 넘어진 것으로 본다.
#   정상 보행 z≈0.6~0.75, 전복/배깔기 z≈0.27 → 0.4로 깔끔히 갈린다.
TOPPLE_Z = 0.4
TOPPLE_PATIENCE = 10  # z<TOPPLE_Z가 이만큼 연속되면 '못 일어남' → 에피소드 조기 종료

# 조향(헤딩 제어): 개미의 '실제 진행방향'(몸통 속도)을 목표 쪽으로 돌린다.
TURN_GAIN = 0.9     # 헤딩오차(rad) → turn 오프셋 (탐색 결과 강한 조향이 유리)
TURN_MAX = 0.5      # turn 한계 (너무 크면 전진속도 급감)
TURN_SIGN = -1.0    # 부호: turn>0이면 진행방향이 시계방향으로 → 실측으로 결정
VEL_EMA = 0.2       # 속도방향 저역통과(EMA) 계수 (걸음새 진동 평활화)
MIN_SPEED = 0.05    # 이보다 느리면 진행방향 불확실 → 조향 보류

# 웨이포인트: 가운데 벽(x=0, y0~6)을 오른쪽으로 우회 → 위로 → 목표
#   (0,0) → (2.5, 0) → (2.5, 6.3) → GOAL(0,6)
WAYPOINTS = [
    np.array([2.5, 0.0]),
    np.array([2.5, 6.3]),
    np.asarray(GOAL_POS, dtype=float),
]
WP_REACH = 0.8      # 이 거리 안에 들면 다음 웨이포인트로

# 다리 정의: (이름, hip_idx, ank_idx, gait_phase, ank_sign, hip_sign)
#   gait_phase : 대각선 트로트 — (FL,BR) 위상 0, (FR,BL) 위상 π
#   ank_sign   : ankle 유효범위 부호 (FL,BR=+ / FR,BL=-)
#   hip_sign   : 스핀 상쇄용 hip 부호 (탐색 결과 FL,FR,BL,BR = +,-,+,+)
LEGS = [
    ("front_left",  2, 3, 0.0,    +1.0, +1.0),
    ("front_right", 4, 5, np.pi,  -1.0, -1.0),
    ("back_left",   6, 7, np.pi,  -1.0, +1.0),
    ("right_back",  0, 1, 0.0,    +1.0, +1.0),
]


def _yaw_from_quat(q):
    """쿼터니언(qw,qx,qy,qz) → yaw(라디안). 몸통이 world에서 어느 방향을 보는지."""
    w, x, y, z = q
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap(a):
    """각도를 [-π, π]로 정규화."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class ScriptedExpert:
    """
    규칙 기반 보행 전문가.

    mode:
      "forward"  — A단계: 조향/웨이포인트 없이 그냥 직진 트로트 (걸음새 튜닝용)
      "heading"  — B단계: 목표(GOAL) 한 점을 향해 방향 잡고 걷기
      "waypoint" — C단계: 벽을 우회하는 웨이포인트를 따라 목표까지 (최종)
    """

    def __init__(self, mode="waypoint", freq=FREQ, hip_amp=HIP_AMP,
                 ank_bias=ANK_BIAS, ank_amp=ANK_AMP, ank_lag=ANK_LAG,
                 turn_gain=TURN_GAIN, turn_max=TURN_MAX, turn_sign=TURN_SIGN,
                 warmup=WARMUP_STEPS, noise_std=0.1, seed=None):
        self.mode = mode
        self.freq = freq
        self.hip_amp = hip_amp
        self.ank_bias = ank_bias
        self.ank_amp = ank_amp
        self.ank_lag = ank_lag
        self.turn_gain = turn_gain
        self.turn_max = turn_max
        self.turn_sign = turn_sign
        self.warmup = warmup
        self.noise_std = noise_std
        self.heading_target = np.asarray(GOAL_POS, dtype=float)
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        """에피소드 시작 시 호출 — 걸음 위상 시계/웨이포인트/속도EMA/워밍업 초기화."""
        self.t = 0       # 0부터 다시 세므로 워밍업 램프도 매 에피소드 처음부터
        self.wp_idx = 0
        self.vel_ema = np.zeros(2)

    def _current_target(self, pos):
        """waypoint 모드: 현재 위치 기준으로 따라갈 목표점을 고르고 도달 시 다음으로."""
        if self.mode == "waypoint":
            if (self.wp_idx < len(WAYPOINTS) - 1 and
                    np.linalg.norm(WAYPOINTS[self.wp_idx] - pos) < WP_REACH):
                self.wp_idx += 1
            return WAYPOINTS[self.wp_idx]
        return self.heading_target

    def act(self, obs):
        """관측 하나 → 행동(8차원, [-1,1])."""
        pos = obs[0:2]
        vel = obs[15:17]                       # 몸통 선속도 (world x,y)
        # 진행방향은 걸음새 때문에 매 스텝 출렁임 → EMA로 평활화
        self.vel_ema = (1.0 - VEL_EMA) * self.vel_ema + VEL_EMA * vel

        # 1) 조향: '실제 진행방향'을 목표 쪽으로 돌리는 turn 오프셋
        turn = 0.0
        if self.mode != "forward":
            target = self._current_target(pos)
            desired = np.arctan2(target[1] - pos[1], target[0] - pos[0])
            speed = np.linalg.norm(self.vel_ema)
            if speed > MIN_SPEED:
                travel = np.arctan2(self.vel_ema[1], self.vel_ema[0])
                err = _wrap(desired - travel)
                turn = float(np.clip(self.turn_sign * self.turn_gain * err,
                                     -self.turn_max, self.turn_max))

        # 2) 대각선 트로트 걸음새 + 조향 오프셋
        #    워밍업: 초반엔 진동/조향을 약하게 시작(ramp 0→1)해 시작 전복을 줄인다.
        ramp = min(1.0, self.t / self.warmup) if self.warmup > 0 else 1.0
        phi = 2.0 * np.pi * self.freq * DT * self.t
        action = np.zeros(8, dtype=np.float32)
        for _name, hip, ank, phase, ank_sign, hip_sign in LEGS:
            a = phi + phase
            # hip: 앞뒤 스윙(추진) + 공통 turn(조향) — 둘 다 램프 적용
            action[hip] = ramp * (hip_sign * self.hip_amp * np.sin(a) + turn)
            # ankle: 펴짐 바이어스(지지, 즉시) + 진동(스텝, 램프)
            action[ank] = ank_sign * (self.ank_bias
                                      + ramp * self.ank_amp * np.sin(a + self.ank_lag))

        # 3) 데이터 다양성용 노이즈 (preview에선 noise_std=0으로 깔끔히 본다)
        if self.noise_std > 0:
            action = action + self.rng.normal(0.0, self.noise_std, size=8)

        self.t += 1
        return np.clip(action, -1.0, 1.0).astype(np.float32)


def collect_demonstrations(env, n_episodes=200, max_steps=1000,
                           mode="waypoint", noise_std=0.05):
    """
    전문가 정책으로 n_episodes 만큼 미로를 풀며 (obs, action) 쌍을 모은다.

    ★ 전복 인지(topple-aware) 수집:
      open-loop 걸음새는 일부 시작조건에서 초반에 넘어져 못 일어난다(z≈0.27).
      넘어진 상태의 다리 버둥거림은 BC 학습에 독(毒)이므로,
      - 몸통이 서 있는(z≥TOPPLE_Z) 스텝의 (obs,action)만 저장하고,
      - z<TOPPLE_Z가 TOPPLE_PATIENCE 연속이면 '못 일어남'으로 보고 에피소드 조기 종료.
      → 데이터셋에는 '똑바로 서서 움직이는' 깨끗한 전이만 남는다.

    반환: dict {
        "observations": (N, 109) float32,   # 서 있는 상태의 전이만
        "actions":      (N, 8)   float32,
        "stats":        성공률/평균스텝/평균보상/총전이수/전복률/저장된전이수
    }
    """
    expert = ScriptedExpert(mode=mode, noise_std=noise_std, seed=0)

    all_obs, all_actions = [], []
    successes = 0
    toppled_eps = 0
    episode_steps, episode_rewards = [], []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)     # 에피소드마다 시드 → 재현 + 다양성(reset 노이즈)
        expert.reset()
        done = False
        steps = 0
        ep_reward = 0.0
        reached = False
        low_streak = 0                     # z<TOPPLE_Z 연속 카운트
        toppled = False

        while not done and steps < max_steps:
            action = expert.act(obs)

            upright = obs[2] >= TOPPLE_Z    # obs[2] = 몸통 높이 z
            if upright:
                all_obs.append(obs.copy())  # 서 있는 전이만 저장
                all_actions.append(action.copy())
                low_streak = 0
            else:
                low_streak += 1

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1

            if info.get("reached_goal", False):
                reached = True
            if low_streak >= TOPPLE_PATIENCE:   # 넘어져서 회복 못함 → 조기 종료
                toppled = True
                break
            done = terminated or truncated

        if reached:
            successes += 1
        if toppled:
            toppled_eps += 1
        episode_steps.append(steps)
        episode_rewards.append(ep_reward)

        if (ep + 1) % 50 == 0:
            print(f"  {ep + 1}/{n_episodes} 에피소드 완료... "
                  f"(누적 성공 {successes}, 전복 {toppled_eps}, 저장 전이 {len(all_obs)})")

    stats = {
        "success_rate": successes / n_episodes,
        "topple_rate": toppled_eps / n_episodes,
        "avg_steps": float(np.mean(episode_steps)),
        "avg_reward": float(np.mean(episode_rewards)),
        "total_transitions": len(all_obs),
    }
    return {
        "observations": np.array(all_obs, dtype=np.float32),
        "actions": np.array(all_actions, dtype=np.float32),
        "stats": stats,
    }

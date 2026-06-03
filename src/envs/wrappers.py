"""환경 래퍼 모음."""
import numpy as np
import gymnasium as gym


class FixedNormalizeObs(gym.ObservationWrapper):
    """
    BC와 '동일한 고정 통계'로 관측을 정규화한다.

    ⚠️ 왜 필요한가:
      BCPolicy는 forward() 안에서 (obs - mean) / (std + 1e-6)로 정규화한 뒤
      신경망에 넣는다. 그 가중치를 PPO로 warm-start하면서 환경이 '날것 obs'를
      주면 스케일이 안 맞아 warm-start가 무의미해진다. 이 래퍼로 PPO 환경도
      BC와 똑같이 정규화한다.

    ⚠️ VecNormalize와 다른 점:
      VecNormalize는 자기만의 running 통계를 '새로' 학습한다. 여기서는 BC가
      학습 때 쓴 '고정' 통계(obs_norm_stats.npz)를 그대로 적용한다.

    ⚠️ 재사용:
      나중에 학습된 PPO 정책을 평가/배포할 때도 반드시 같은 통계로 이 래퍼를
      씌워야 한다. 안 그러면 같은 버그가 재발한다.
    """

    def __init__(self, env, mean, std, clip=10.0):
        super().__init__(env)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.clip = clip                       # 분포 밖 관측 폭발 방지 (SB3 VecNormalize 기본값과 동일)
        self.observation_space = gym.spaces.Box(
            low=-clip, high=clip,
            shape=env.observation_space.shape, dtype=np.float32,
        )

    def observation(self, obs):
        obs = (obs - self.mean) / (self.std + 1e-6)
        return np.clip(obs, -self.clip, self.clip).astype(np.float32)

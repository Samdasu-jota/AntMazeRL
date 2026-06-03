"""
behavior_cloning.py — 전문가 시범을 흉내내는 학생 신경망 (Behavior Cloning).

핵심 아이디어:
  2.1에서 모은 (관측 obs, 행동 action) 16만 쌍을 보고,
  "이 관측이 들어오면 이 행동을 내라"를 신경망(MLP)에게 지도학습시킨다.
  정답(전문가 행동)과 예측의 차이(MSE)를 줄이는 단순 회귀 문제.

  입력: 관측 109차원 → 출력: 행동 8차원 (tanh로 [-1,1] 제한)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split


def get_device():
    """GPU(CUDA) 있으면 GPU, 없으면 CPU.

    MPS(Apple Silicon)는 일부러 제외한다: 이 소형 MLP(109→256³→8)는 매 배치
    host↔device 전송 오버헤드가 연산 이득을 넘어 CPU가 더 빠르고, MPS는
    간헐적 부정확/미지원 op 에러 위험이 있다.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")   # 소형 MLP는 MPS보다 CPU가 빠름


class BCPolicy(nn.Module):
    """
    관측(obs) → 행동(action) 매핑 MLP.
    3개 은닉층 + ReLU, 마지막에 tanh로 행동을 [-1,1]로 제한.

    관측 정규화를 모델 안에 내장:
      obs는 차원마다 스케일이 천차만별(위치 vs 접촉력 vs 관절각).
      정규화 안 하면 큰 값 차원이 학습을 지배한다. mean/std를 buffer로
      저장해 forward에서 자동 정규화 → 추론 시 날것 obs를 그냥 넣으면 됨.
    """

    def __init__(self, obs_dim=109, act_dim=8, hidden_dim=256,
                 obs_mean=None, obs_std=None):
        super().__init__()
        # 정규화 통계 (buffer = 학습 안 되지만 모델과 함께 저장/로드됨)
        if obs_mean is None:
            obs_mean = torch.zeros(obs_dim)
        if obs_std is None:
            obs_std = torch.ones(obs_dim)
        self.register_buffer("obs_mean", torch.as_tensor(obs_mean, dtype=torch.float32))
        self.register_buffer("obs_std", torch.as_tensor(obs_std, dtype=torch.float32))

        # 신경망 본체: 109 → 256 → 256 → 256 → 8
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),     # 입력 → 은닉1
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  # 은닉1 → 은닉2
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  # 은닉2 → 은닉3
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),     # 은닉3 → 출력
            nn.Tanh(),                          # 행동 [-1,1]로 제한
        )

    def forward(self, obs):
        # 관측 정규화 후 신경망 통과 (1e-6은 0으로 나누기 방지)
        obs = (obs - self.obs_mean) / (self.obs_std + 1e-6)
        return self.net(obs)


def load_data(data_path, val_split, batch_size, seed):
    """npz 로드 → 정규화 통계 계산 → train/val DataLoader 생성."""
    data = np.load(data_path)
    obs = torch.as_tensor(data["observations"], dtype=torch.float32)
    act = torch.as_tensor(data["actions"], dtype=torch.float32)

    # 각 차원별 평균/표준편차 (정규화용)
    obs_mean = obs.mean(dim=0)
    obs_std = obs.std(dim=0)

    # 전체를 (관측, 행동) 묶음으로 만들고 train/val로 분할
    dataset = TensorDataset(obs, act)
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    gen = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=gen)

    # DataLoader: 데이터를 batch_size씩 잘라서 공급. 학습용은 매번 섞음(shuffle).
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, obs_mean, obs_std


def train_bc(config):
    """BC 학습 메인 루프. 반환: (train_losses, val_losses) 리스트."""
    device = get_device()
    print(f"학습 장치: {device}")

    train_loader, val_loader, obs_mean, obs_std = load_data(
        config["data_path"], config["val_split"],
        config["batch_size"], config["seed"],
    )

    # 정규화 통계를 별도 파일로도 저장 (3단계 PPO warm-start 시 SB3에 넘기기 편함)
    if config.get("norm_out"):
        np.savez(config["norm_out"],
                 obs_mean=obs_mean.cpu().numpy(),
                 obs_std=obs_std.cpu().numpy())
        print(f"정규화 통계 저장: {config['norm_out']}")

    # 모델 + 옵티마이저 + 손실함수
    model = BCPolicy(hidden_dim=config["hidden_dim"],
                     obs_mean=obs_mean, obs_std=obs_std).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    loss_fn = nn.MSELoss()   # 예측 행동 vs 전문가 행동 차이의 제곱 평균

    train_losses, val_losses = [], []
    best_val = float("inf")   # 지금까지 최저 검증 손실 (이걸 갱신할 때만 저장)

    for epoch in range(config["epochs"]):
        # ── 학습 단계 ──
        model.train()
        epoch_train = 0.0
        for obs_b, act_b in train_loader:
            obs_b, act_b = obs_b.to(device), act_b.to(device)
            pred = model(obs_b)                 # 예측 행동
            loss = loss_fn(pred, act_b)          # 전문가 행동과의 차이
            optimizer.zero_grad()                # 이전 기울기 초기화
            loss.backward()                      # 역전파 (기울기 계산)
            optimizer.step()                     # 가중치 업데이트
            epoch_train += loss.item() * len(obs_b)
        epoch_train /= len(train_loader.dataset)

        # ── 검증 단계 (학습 안 함, 채점만) ──
        model.eval()
        epoch_val = 0.0
        with torch.no_grad():                    # 기울기 계산 끔 (빠름)
            for obs_b, act_b in val_loader:
                obs_b, act_b = obs_b.to(device), act_b.to(device)
                epoch_val += loss_fn(model(obs_b), act_b).item() * len(obs_b)
        epoch_val /= len(val_loader.dataset)

        train_losses.append(epoch_train)
        val_losses.append(epoch_val)

        # 검증 손실이 최저면 모델 저장 (과적합 전 최적 상태 보존)
        if epoch_val < best_val:
            best_val = epoch_val
            torch.save(model.state_dict(), config["model_out"])

        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch + 1:3d}/{config['epochs']} | "
                  f"train {epoch_train:.4f} | val {epoch_val:.4f}")

    return train_losses, val_losses


def evaluate_policy(env, model_path, n_episodes=20, hidden_dim=256):
    """학습된 BC 정책을 환경에서 평가 → (평균 보상, 성공률)."""
    device = get_device()
    model = BCPolicy(hidden_dim=hidden_dim)
    # 저장된 가중치 로드 (정규화 buffer도 함께 들어옴)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    rewards, successes = [], 0
    for ep in range(n_episodes):
        obs, info = env.reset(seed=1000 + ep)   # 학습과 다른 시드로 평가
        done = False
        ep_reward = 0.0
        reached = False
        while not done:
            # obs를 텐서로 → 모델 통과 → 행동 추출
            obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                    device=device).unsqueeze(0)
            with torch.no_grad():
                action = model(obs_t).cpu().numpy()[0]
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            if info.get("reached_goal", False):
                reached = True
            done = terminated or truncated
        rewards.append(ep_reward)
        if reached:
            successes += 1

    return float(np.mean(rewards)), successes / n_episodes

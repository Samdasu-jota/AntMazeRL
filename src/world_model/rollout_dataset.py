"""
Phase 4.1 — world_model_rollouts.npz → PyTorch Dataset.

입력 x = [정규화 state | 정규화 action] (117차원), 타겟 y = 다음상태(기본은 'delta'
= next_state - state) 정규화. World Model은 작은 보정(delta)을 배우는 게 절대 next_state보다
조건수가 좋다(항등사상이 강한 prior).

정규화 통계는 여기서 'train split에서만' 적합(누수 방지)하고 data/world_model_norm.npz에
저장한다 — data/obs_norm_stats.npz(BC/PPO 정책 입력 통계)와는 분포가 달라 재사용하면 안 된다.
정책 입력용 ±10 클립은 여기 적용하지 않는다(월드모델은 실제 분포를 봐야 함).

split은 '전이' 단위가 아니라 '에피소드' 단위로 나눈다 — 인접 전이는 거의 동일해 전이단위
분할은 누수가 된다. 같은 (val_frac, split_seed)면 train/val이 같은 분할에 합의한다.

비-Markov 주의: wp_switch=True인 전이는 obs[107:109](목표 상대벡터)가 점프한다(웨이포인트
전환). __getitem__이 wp_switch를 함께 반환하므로 4.2가 그 타겟을 mask/drop/재계산할 수 있다.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

NORM_PATH = "data/world_model_norm.npz"


def _episode_split(episode_ids, val_frac, split_seed):
    """에피소드 인덱스를 train/val로 결정적 분할 → 전이 boolean mask 두 개 반환."""
    E = int(episode_ids.max()) + 1
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(E)
    n_val = max(1, int(round(val_frac * E)))
    val_eps = set(perm[:n_val].tolist())
    is_val = np.array([eid in val_eps for eid in episode_ids], dtype=bool)
    return ~is_val, is_val   # train_mask, val_mask


class RolloutDataset(Dataset):
    def __init__(self, npz_path="data/world_model_rollouts.npz", split="train",
                 val_frac=0.1, split_seed=0, predict="delta", include_reward=False,
                 norm_path=NORM_PATH, save_norm=True):
        assert split in ("train", "val", "all")
        assert predict in ("delta", "absolute")
        data = np.load(npz_path, allow_pickle=True)
        states = data["states"].astype(np.float32)
        actions = data["actions"].astype(np.float32)
        next_states = data["next_states"].astype(np.float32)
        rewards = data["rewards"].astype(np.float32)
        wp_switch = data["wp_switch"].astype(bool)
        episode_ids = data["episode_ids"]
        deltas = next_states - states

        train_mask, val_mask = _episode_split(episode_ids, val_frac, split_seed)

        # 통계는 항상 'train 부분집합'에서 적합(어느 split 인스턴스든 동일) → 누수 0
        def stats(arr):
            tr = arr[train_mask]
            return tr.mean(0).astype(np.float32), (tr.std(0) + 1e-6).astype(np.float32)
        self.state_mean, self.state_std = stats(states)
        self.action_mean, self.action_std = stats(actions)
        self.delta_mean, self.delta_std = stats(deltas)
        if save_norm:
            np.savez(norm_path,
                     state_mean=self.state_mean, state_std=self.state_std,
                     action_mean=self.action_mean, action_std=self.action_std,
                     delta_mean=self.delta_mean, delta_std=self.delta_std)

        m = {"train": train_mask, "val": val_mask,
             "all": np.ones(len(states), dtype=bool)}[split]
        self.states, self.actions = states[m], actions[m]
        self.next_states, self.rewards = next_states[m], rewards[m]
        self.deltas, self.wp_switch = deltas[m], wp_switch[m]
        self.dones = data["dones"].astype(np.float32)[m]
        self.episode_ids = episode_ids[m]
        self.predict, self.include_reward = predict, include_reward
        self.state_dim, self.action_dim = states.shape[1], actions.shape[1]

    def __len__(self):
        return len(self.states)

    def __getitem__(self, i):
        xs = (self.states[i] - self.state_mean) / self.state_std
        xa = (self.actions[i] - self.action_mean) / self.action_std
        x = np.concatenate([xs, xa]).astype(np.float32)
        if self.predict == "delta":
            y = (self.deltas[i] - self.delta_mean) / self.delta_std
        else:
            y = (self.next_states[i] - self.state_mean) / self.state_std
        out = dict(
            x=torch.from_numpy(x),
            y_next=torch.from_numpy(y.astype(np.float32)),
            done=torch.tensor(self.dones[i]),
            wp_switch=torch.tensor(bool(self.wp_switch[i])),
            episode_id=torch.tensor(int(self.episode_ids[i])),
        )
        if self.include_reward:
            out["y_reward"] = torch.tensor(np.float32(self.rewards[i]))
        return out

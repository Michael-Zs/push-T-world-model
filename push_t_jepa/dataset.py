"""从确定性 Push-T 环境采集 JEPA 训练样本。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import EnvConfig
from .env import PushTEnv


@dataclass(frozen=True)
class Trajectory:
    """一条图像和动作对齐的环境轨迹。"""

    images: torch.Tensor
    actions: torch.Tensor


def collect_trajectories(
    env_config: EnvConfig | None = None,
    trajectories: int = 5_000,
    steps: int = 12,
    seed: int = 7,
    guided_fraction: float = 0.7,
    progress: Callable[[int, int], None] | None = None,
) -> list[Trajectory]:
    """采集由接触导向策略与随机探索混合组成的可重复轨迹。"""
    if trajectories <= 0:
        raise ValueError("轨迹数量必须为正数")
    if steps <= 0:
        raise ValueError("每条轨迹步数必须为正数")
    if not 0.0 <= guided_fraction <= 1.0:
        raise ValueError("导向动作比例必须位于 0 到 1 之间")
    rng = np.random.default_rng(seed)
    result: list[Trajectory] = []
    for _ in range(trajectories):
        env = PushTEnv(config=env_config, seed=int(rng.integers(0, 2**31 - 1)))
        images = [env.reset()]
        actions: list[np.ndarray] = []
        for _ in range(steps):
            if rng.random() < guided_fraction:
                delta = env.state.object_position - env.state.pusher
                distance = float(np.linalg.norm(delta))
                direction = delta / max(distance, 1e-6)
                action = direction * min(1.0, distance / env.config.max_action)
                action = action + rng.normal(0.0, 0.18, size=2)
                action = np.clip(action, -1.0, 1.0).astype(np.float32)
            else:
                action = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
            image, _ = env.step(action)
            actions.append(action)
            images.append(image)
        result.append(
            Trajectory(
                images=torch.from_numpy(np.stack(images)).to(torch.uint8),
                actions=torch.from_numpy(np.stack(actions)).to(torch.float32),
            )
        )
        if progress is not None:
            progress(len(result), trajectories)
    return result


class PushTJEPADataset(Dataset[dict[str, torch.Tensor]]):
    """将轨迹转换成当前图像、未来动作和目标图像样本。"""

    def __init__(self, trajectories: list[Trajectory], horizon: int) -> None:
        if horizon <= 0:
            raise ValueError("预测步数必须为正数")
        self._trajectories = trajectories
        self._horizon = horizon
        self._indices: list[tuple[int, int]] = []
        for trajectory_index, trajectory in enumerate(trajectories):
            if trajectory.images.ndim != 4 or trajectory.actions.ndim != 2:
                raise ValueError("轨迹图像或动作张量形状无效")
            if trajectory.images.shape[0] != trajectory.actions.shape[0] + 1:
                raise ValueError("轨迹图像数量必须比动作数量多一")
            for time_index in range(trajectory.actions.shape[0] - horizon + 1):
                self._indices.append((trajectory_index, time_index))
        if not self._indices:
            raise ValueError("轨迹长度不足以构造指定预测步数的样本")

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        trajectory_index, time_index = self._indices[index]
        trajectory = self._trajectories[trajectory_index]
        future_index = time_index + self._horizon
        return {
            "image": self._normalize_image(trajectory.images[time_index]),
            "actions": trajectory.actions[time_index:future_index].clone(),
            "future_image": self._normalize_image(trajectory.images[future_index]),
        }

    @staticmethod
    def _normalize_image(image: torch.Tensor) -> torch.Tensor:
        return image.permute(2, 0, 1).to(torch.float32).div(255.0)

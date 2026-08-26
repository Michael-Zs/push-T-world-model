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


@dataclass(frozen=True)
class CollectionStats:
    effective_step_rate: float
    mean_translation: float
    mean_rotation: float


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


def collect_trajectories_with_stats(
    env_config: EnvConfig | None = None,
    trajectories: int = 5_000,
    steps: int = 64,
    seed: int = 7,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[list[Trajectory], CollectionStats]:
    """采集平衡的无接触、平移与偏心旋转轨迹，并返回有效样本统计。"""
    if trajectories <= 0 or steps <= 0:
        raise ValueError("轨迹数量和每条轨迹步数必须为正数")
    rng = np.random.default_rng(seed)
    result: list[Trajectory] = []
    effective = 0; total = 0; translation = 0.0; rotation = 0.0
    for index in range(trajectories):
        env = PushTEnv(config=env_config, seed=int(rng.integers(0, 2**31 - 1)))
        env.reset()
        mode = index % 5
        direction = rng.normal(size=2).astype(np.float32); direction /= max(float(np.linalg.norm(direction)), 1e-6)
        perpendicular = np.array([-direction[1], direction[0]], dtype=np.float32)
        if mode:
            offset = 0.0 if mode in (1, 2) else float(rng.choice([-1.0, 1.0])) * 0.065
            object_position = np.array([0.5, 0.5], dtype=np.float32)
            pusher = object_position - direction * 0.17 + perpendicular * offset
            env.set_state(pusher, object_position, float(rng.uniform(-np.pi, np.pi)))
        images = [env.render()]; actions: list[np.ndarray] = []
        for _ in range(steps):
            before = env.state
            action = rng.uniform(-1.0, 1.0, size=2).astype(np.float32) if mode == 0 else np.clip(direction + rng.normal(0.0, 0.08, 2), -1, 1).astype(np.float32)
            image, after = env.step(action)
            move = float(np.linalg.norm(after.object_position - before.object_position)); turn = abs(after.object_angle - before.object_angle)
            effective += move > 1e-4 or turn > 1e-4; total += 1; translation += move; rotation += turn
            images.append(image); actions.append(action)
        result.append(Trajectory(torch.from_numpy(np.stack(images)).to(torch.uint8), torch.from_numpy(np.stack(actions)).to(torch.float32)))
        if progress is not None: progress(index + 1, trajectories)
    return result, CollectionStats(effective / total, translation / trajectories, rotation / trajectories)


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

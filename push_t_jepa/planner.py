"""在 JEPA embedding 空间中工作的 CEM 滚动规划器。"""

from __future__ import annotations

import numpy as np
import torch

from .config import CEMConfig
from .env import PushTEnv
from .model import JEPAModel


class CEMPlanner:
    """通过交叉熵方法搜索使终点接近目标 embedding 的动作序列。"""

    def __init__(self, model: JEPAModel, config: CEMConfig | None = None, seed: int = 7) -> None:
        self.model = model
        self.config = config or CEMConfig()
        if self.config.horizon % model.config.action_horizon != 0:
            raise ValueError("CEM 规划步数必须是模型动作预测步数的整数倍")
        self._rng = np.random.default_rng(seed)

    @torch.no_grad()
    def plan(self, current_image: np.ndarray, target_image: np.ndarray) -> np.ndarray:
        """返回 CEM 优化后的合法二维动作序列。"""
        device = next(self.model.parameters()).device
        current = self._image_tensor(current_image, device)
        target = self._image_tensor(target_image, device)
        previous_mode = self.model.training
        self.model.eval()
        target_pose = self.model.predict_pose(self.model.encode_target(target))[:, 2:]
        context = self.model.encode_context(current).repeat(self.config.population, 1, 1, 1)
        mean = np.zeros((self.config.horizon, 2), dtype=np.float32)
        std = np.ones_like(mean)
        best = mean.copy()
        for _ in range(self.config.iterations):
            candidates = np.clip(
                self._rng.normal(mean, std, size=(self.config.population, self.config.horizon, 2)),
                -1.0,
                1.0,
            ).astype(np.float32)
            actions = torch.from_numpy(candidates).to(device)
            terminal = self._predict_terminal_embeddings(context, actions)
            terminal_pose = self.model.predict_pose(terminal)[:, 2:]
            embedding_cost = (terminal_pose - target_pose).square().sum(dim=1)
            action_cost = 1e-3 * actions.square().sum(dim=(1, 2))
            costs = (embedding_cost + action_cost).cpu().numpy()
            elite_indices = np.argpartition(costs, self.config.elite_count - 1)[: self.config.elite_count]
            elite = candidates[elite_indices]
            mean = elite.mean(axis=0)
            std = np.maximum(elite.std(axis=0), 0.05)
            best = candidates[int(costs.argmin())]
        if previous_mode:
            self.model.train()
        return np.clip(best, -1.0, 1.0)

    def rollout_replan(self, env: PushTEnv, target_image: np.ndarray, steps: int) -> list[np.ndarray]:
        """每次只执行最优序列第一步，并记录执行得到的观察。"""
        if steps < 0:
            raise ValueError("执行步数不能为负数")
        observations = [env.render()]
        for _ in range(steps):
            action = self.next_action(observations[-1], target_image)
            observation, _ = env.step(action)
            observations.append(observation)
        return observations

    def next_action(self, current_image: np.ndarray, target_image: np.ndarray) -> np.ndarray:
        """尚未接触 T 时先靠近，接触后执行 CEM 的首个动作。"""
        approach = self.approach_action(current_image)
        return approach if approach is not None else self.plan(current_image, target_image)[0]

    def approach_action(self, image: np.ndarray) -> np.ndarray | None:
        """从渲染图像提取蓝色推杆与深色 T，并返回朝最近接触点移动的动作。"""
        array = np.asarray(image)
        size = self.model.config.image_size
        if array.shape != (size, size, 3):
            raise ValueError(f"图像必须是形状为 [{size}, {size}, 3] 的 RGB 数组")
        red, green, blue = (array[..., index].astype(np.int16) for index in range(3))
        pusher_mask = (blue > green + 45) & (blue > red + 70)
        object_mask = np.max(array, axis=2) < 100
        if not np.any(pusher_mask) or not np.any(object_mask):
            return None
        pusher_yx = np.argwhere(pusher_mask).mean(axis=0)
        object_yx = np.argwhere(object_mask)
        offsets = object_yx - pusher_yx
        nearest = offsets[np.argmin(np.square(offsets).sum(axis=1))]
        distance = float(np.linalg.norm(nearest))
        pusher_radius = max(2.0, 0.045 * (size - 1))
        if distance <= pusher_radius + 1.5:
            return None
        direction_yx = nearest / distance
        return np.asarray([direction_yx[1], direction_yx[0]], dtype=np.float32)

    def _predict_terminal_embeddings(self, context: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        latent = context
        step = self.model.config.action_horizon
        for start in range(0, self.config.horizon, step):
            latent = self.model.predict_from_context(latent, actions[:, start : start + step])
        return latent

    def _image_tensor(self, image: np.ndarray, device: torch.device) -> torch.Tensor:
        array = np.asarray(image)
        size = self.model.config.image_size
        if array.shape != (size, size, 3):
            raise ValueError(f"图像必须是形状为 [{size}, {size}, 3] 的 RGB 数组")
        return torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32).div(255.0)

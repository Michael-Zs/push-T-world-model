"""无需 GUI 的确定性二维 Push-T 近似环境。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .config import EnvConfig


@dataclass
class PushTState:
    """场景中推杆和 T 物体的平面位姿。"""

    pusher: np.ndarray
    object_position: np.ndarray
    object_angle: float

    def copy(self) -> "PushTState":
        return PushTState(self.pusher.copy(), self.object_position.copy(), self.object_angle)


class PushTEnv:
    """以近似接触规则推动 T 形物体的确定性环境。"""

    _PUSHER_RADIUS = 0.045
    _OBJECT_REACH = 0.13

    def __init__(self, config: EnvConfig | None = None, seed: int = 7) -> None:
        self.config = config or EnvConfig()
        self._rng = np.random.default_rng(seed)
        self._state = PushTState(
            pusher=np.array([0.2, 0.5], dtype=np.float32),
            object_position=np.array([0.5, 0.5], dtype=np.float32),
            object_angle=0.0,
        )

    @property
    def state(self) -> PushTState:
        """返回状态副本，避免调用方修改环境内部状态。"""
        return self._state.copy()

    def reset(self, seed: int | None = None) -> np.ndarray:
        """重置为由种子决定的非重叠初始位姿并返回图像。"""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        object_position = self._rng.uniform(0.35, 0.65, size=2).astype(np.float32)
        pusher = self._rng.uniform(0.12, 0.30, size=2).astype(np.float32)
        self._state = PushTState(
            pusher=pusher,
            object_position=object_position,
            object_angle=float(self._rng.uniform(-np.pi, np.pi)),
        )
        return self.render()

    def set_state(self, pusher: np.ndarray, object_position: np.ndarray, object_angle: float) -> None:
        """设置状态；该接口用于确定性实验和测试。"""
        self._state = PushTState(
            pusher=self._clip_position(self._position(pusher, "推杆位置")),
            object_position=self._clip_position(self._position(object_position, "物体位置")),
            object_angle=float(object_angle),
        )

    def step(self, action: np.ndarray) -> tuple[np.ndarray, PushTState]:
        """执行归一化二维动作，返回新图像和只读状态副本。"""
        action_array = np.asarray(action, dtype=np.float32)
        if action_array.shape != (2,) or not np.all(np.isfinite(action_array)):
            raise ValueError("动作必须是包含两个有限数值的二维数组")
        action_array = np.clip(action_array, -1.0, 1.0)
        displacement = action_array * self.config.max_action
        old_pusher = self._state.pusher.copy()
        self._state.pusher = self._clip_position(old_pusher + displacement)

        distance = np.linalg.norm(self._state.pusher - self._state.object_position)
        movement_norm = float(np.linalg.norm(displacement))
        if distance <= self._OBJECT_REACH and movement_norm > 1e-8:
            direction = displacement / movement_norm
            contact_offset = old_pusher - self._state.object_position
            self._state.object_position = self._clip_position(
                self._state.object_position + direction * movement_norm * 0.72
            )
            torque = float(contact_offset[0] * direction[1] - contact_offset[1] * direction[0])
            self._state.object_angle = float((self._state.object_angle + torque * 2.0 + np.pi) % (2 * np.pi) - np.pi)
        return self.render(), self.state

    def render(self) -> np.ndarray:
        """使用 Pillow 渲染当前场景为 RGB 图像。"""
        size = self.config.image_size
        image = Image.new("RGB", (size, size), (245, 247, 250))
        draw = ImageDraw.Draw(image)
        draw.rectangle((1, 1, size - 2, size - 2), outline=(145, 155, 170), width=1)
        polygon = [self._to_pixel(point) for point in self._t_vertices()]
        draw.polygon(polygon, fill=(50, 55, 65), outline=(20, 20, 25))
        x, y = self._to_pixel(self._state.pusher)
        radius = max(2, round(self._PUSHER_RADIUS * (size - 1)))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(40, 120, 220), outline=(15, 70, 150))
        return np.asarray(image, dtype=np.uint8)

    def _position(self, value: np.ndarray, name: str) -> np.ndarray:
        position = np.asarray(value, dtype=np.float32)
        if position.shape != (2,) or not np.all(np.isfinite(position)):
            raise ValueError(f"{name}必须是包含两个有限数值的二维数组")
        return position

    def _clip_position(self, position: np.ndarray) -> np.ndarray:
        return np.clip(position, 0.0, self.config.arena_size).astype(np.float32)

    def _t_vertices(self) -> np.ndarray:
        local = np.array(
            [
                [-0.12, -0.10], [0.12, -0.10], [0.12, -0.03],
                [0.04, -0.03], [0.04, 0.12], [-0.04, 0.12],
                [-0.04, -0.03], [-0.12, -0.03],
            ],
            dtype=np.float32,
        )
        cosine, sine = np.cos(self._state.object_angle), np.sin(self._state.object_angle)
        rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float32)
        return local @ rotation.T + self._state.object_position

    def _to_pixel(self, point: np.ndarray) -> tuple[int, int]:
        scale = self.config.image_size - 1
        return int(round(float(point[0]) * scale)), int(round(float(point[1]) * scale))

"""由 Pymunk 驱动的确定性二维 Push-T 环境。"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pymunk
from PIL import Image, ImageDraw
from .config import EnvConfig


@dataclass
class PushTState:
    pusher: np.ndarray
    object_position: np.ndarray
    object_angle: float

    def copy(self) -> "PushTState":
        return PushTState(
            self.pusher.copy(), self.object_position.copy(), self.object_angle
        )


class PushTEnv:
    _PUSHER_RADIUS = 0.045

    def __init__(self, config: EnvConfig | None = None, seed: int = 7) -> None:
        self.config = config or EnvConfig()
        self._rng = np.random.default_rng(seed)
        self._build_space()

    def _build_space(self) -> None:
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.space.damping = 0.92
        self.space.iterations = 30
        self.space.collision_slop = 0.0001
        static = self.space.static_body
        size = self.config.arena_size
        self.walls = [
            pymunk.Segment(static, a, b, 0.01)
            for a, b in [
                ((0, 0), (size, 0)),
                ((size, 0), (size, size)),
                ((size, size), (0, size)),
                ((0, size), (0, 0)),
            ]
        ]
        for wall in self.walls:
            wall.friction = 0.9
            wall.elasticity = 0.0
        self.pusher_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.pusher_body.position = (0.2, 0.5)
        self.pusher_shape = pymunk.Circle(self.pusher_body, self._PUSHER_RADIUS)
        self.pusher_shape.friction = 0.7
        self.pusher_shape.elasticity = 0
        mass = 1.0
        moment = pymunk.moment_for_box(mass, (0.24, 0.22))
        self.object_body = pymunk.Body(mass, moment)
        self.object_body.position = (0.5, 0.5)
        self.object_shapes = [
            pymunk.Poly(
                self.object_body,
                [(-0.12, -0.10), (0.12, -0.10), (0.12, -0.03), (-0.12, -0.03)],
            ),
            pymunk.Poly(
                self.object_body,
                [(-0.04, -0.03), (0.04, -0.03), (0.04, 0.12), (-0.04, 0.12)],
            ),
        ]
        for shape in self.object_shapes:
            shape.friction = self.config.object_friction
            shape.elasticity = 0
        self.space.add(
            *self.walls,
            self.pusher_body,
            self.pusher_shape,
            self.object_body,
            *self.object_shapes,
        )

    @property
    def state(self) -> PushTState:
        return PushTState(
            np.array(self.pusher_body.position, dtype=np.float32),
            np.array(self.object_body.position, dtype=np.float32),
            float(self.object_body.angle),
        )

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.set_state(
            self._rng.uniform(0.12, 0.30, 2),
            self._rng.uniform(0.35, 0.65, 2),
            float(self._rng.uniform(-np.pi, np.pi)),
        )
        return self.render()

    def set_state(
        self, pusher: np.ndarray, object_position: np.ndarray, object_angle: float
    ) -> None:
        for value, name in [(pusher, "推杆位置"), (object_position, "物体位置")]:
            if np.asarray(value).shape != (2,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name}必须是包含两个有限数值的二维数组")
        self.pusher_body.position = tuple(np.clip(pusher, 0, self.config.arena_size))
        self.pusher_body.velocity = (0, 0)
        self.object_body.position = tuple(
            np.clip(object_position, 0.13, self.config.arena_size - 0.13)
        )
        self.object_body.angle = float(object_angle)
        self.object_body.velocity = (0, 0)
        self.object_body.angular_velocity = 0
        self.space.reindex_shapes_for_body(self.pusher_body)
        self.space.reindex_shapes_for_body(self.object_body)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, PushTState]:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,) or not np.all(np.isfinite(action)):
            raise ValueError("动作必须是包含两个有限数值的二维数组")
        displacement = np.clip(
            np.clip(action, -1, 1) * self.config.max_action, -0.01, 0.01
        )
        candidate = np.clip(
            np.asarray(self.pusher_body.position) + displacement,
            0.0,
            self.config.arena_size,
        )
        query = min(
            (shape.point_query(tuple(candidate)) for shape in self.object_shapes),
            key=lambda item: item.distance,
        )
        if query.distance <= self._PUSHER_RADIUS:
            normal = np.asarray(query.gradient, dtype=np.float32)
            push_amount = max(0.0, -float(np.dot(displacement, normal)))
            if push_amount > 0.0:
                force = -normal * push_amount
                center = np.asarray(self.object_body.position, dtype=np.float32)
                position = np.clip(center + force, 0.13, self.config.arena_size - 0.13)
                lever = np.asarray(query.point, dtype=np.float32) - center
                torque = float(lever[0] * force[1] - lever[1] * force[0])
                self.object_body.position = tuple(position)
                self.object_body.angle += torque / max(self.object_body.moment, 1e-6)
                self.space.reindex_shapes_for_body(self.object_body)
                query = min(
                    (
                        shape.point_query(tuple(candidate))
                        for shape in self.object_shapes
                    ),
                    key=lambda item: item.distance,
                )
            if query.distance < self._PUSHER_RADIUS:
                candidate = (
                    np.asarray(query.point)
                    + np.asarray(query.gradient) * self._PUSHER_RADIUS
                )
        self.pusher_body.position = tuple(candidate)
        self.object_body.velocity = (0, 0)
        self.object_body.angular_velocity = 0
        self.pusher_body.velocity = (0, 0)
        self.space.reindex_shapes_for_body(self.pusher_body)
        return self.render(), self.state

    def pusher_clearance(self) -> float:
        center = self.pusher_body.position
        distances = [
            shape.point_query(center).distance - self._PUSHER_RADIUS
            for shape in self.object_shapes
        ]
        return float(min(distances))

    def render(self) -> np.ndarray:
        size = self.config.image_size
        image = Image.new("RGB", (size, size), (245, 247, 250))
        draw = ImageDraw.Draw(image)
        draw.rectangle((1, 1, size - 2, size - 2), outline=(145, 155, 170), width=1)
        for shape in self.object_shapes:
            world_vertices = [
                self.object_body.local_to_world(vertex)
                for vertex in shape.get_vertices()
            ]
            draw.polygon(
                [self._pixel(vertex) for vertex in world_vertices],
                fill=(50, 55, 65),
                outline=(20, 20, 25),
            )
        x, y = self._pixel(self.pusher_body.position)
        r = max(2, round(self._PUSHER_RADIUS * (size - 1)))
        draw.ellipse(
            (x - r, y - r, x + r, y + r), fill=(40, 120, 220), outline=(15, 70, 150)
        )
        return np.asarray(image, dtype=np.uint8)

    def _pixel(self, point: object) -> tuple[int, int]:
        size = self.config.image_size - 1
        return int(round(float(point[0]) * size)), int(round(float(point[1]) * size))

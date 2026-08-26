"""实验所需的不可变默认配置。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvConfig:
    """Push-T 环境参数。"""

    image_size: int = 64
    arena_size: float = 1.0
    max_action: float = 0.08
    physics_dt: float = 1 / 240
    physics_substeps: int = 8
    pusher_speed: float = 1.0
    object_friction: float = 0.8

    def __post_init__(self) -> None:
        if self.image_size <= 0 or self.arena_size <= 0:
            raise ValueError("图像尺寸和场地尺寸必须为正数")
        if self.physics_dt <= 0 or self.physics_substeps <= 0 or self.pusher_speed <= 0:
            raise ValueError("物理时间步、子步数和推杆速度必须为正数")
        if self.object_friction < 0:
            raise ValueError("物体摩擦系数不能为负数")


@dataclass(frozen=True)
class ModelConfig:
    """JEPA 模型参数。"""

    embedding_dim: int = 64
    action_horizon: int = 4
    image_size: int = 64
    spatial_size: int = 8

    def __post_init__(self) -> None:
        if self.image_size < 64 or self.image_size & (self.image_size - 1):
            raise ValueError("图像尺寸必须是大于等于 64 的 2 的幂")
        if self.embedding_dim <= 0 or self.action_horizon <= 0:
            raise ValueError("latent 通道数和动作预测步数必须为正数")
        if self.spatial_size != 8:
            raise ValueError("当前空间 latent 尺寸固定为 8")


@dataclass(frozen=True)
class CEMConfig:
    """交叉熵方法规划器参数。"""

    horizon: int = 8
    population: int = 128
    elite_count: int = 16
    iterations: int = 3

    def __post_init__(self) -> None:
        if (
            self.horizon <= 0
            or self.population <= 0
            or self.elite_count <= 0
            or self.iterations <= 0
        ):
            raise ValueError("CEM 参数必须为正数")
        if self.elite_count > self.population:
            raise ValueError("精英数量不能大于候选数量")


@dataclass(frozen=True)
class TrainConfig:
    """训练过程参数。"""

    seed: int = 7
    device: str = "cpu"

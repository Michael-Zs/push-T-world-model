"""实验所需的不可变默认配置。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvConfig:
    """Push-T 环境参数。"""

    image_size: int = 64
    arena_size: float = 1.0
    max_action: float = 0.08


@dataclass(frozen=True)
class ModelConfig:
    """JEPA 模型参数。"""

    embedding_dim: int = 64
    action_horizon: int = 4


@dataclass(frozen=True)
class CEMConfig:
    """交叉熵方法规划器参数。"""

    horizon: int = 8
    population: int = 128
    elite_count: int = 16
    iterations: int = 3

    def __post_init__(self) -> None:
        if self.horizon <= 0 or self.population <= 0 or self.elite_count <= 0 or self.iterations <= 0:
            raise ValueError("CEM 参数必须为正数")
        if self.elite_count > self.population:
            raise ValueError("精英数量不能大于候选数量")


@dataclass(frozen=True)
class TrainConfig:
    """训练过程参数。"""

    seed: int = 7
    device: str = "cpu"

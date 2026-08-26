"""保留空间布局的动作条件 JEPA。"""

from __future__ import annotations

import copy
from collections.abc import Mapping

import torch
from torch import nn

from .config import ModelConfig


class ImageEncoder(nn.Module):
    """将图像编码为固定 8x8 的空间 latent。"""

    def __init__(self, embedding_dim: int, spatial_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((spatial_size, spatial_size)),
            nn.Conv2d(48, embedding_dim, kernel_size=1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.network(image)


class VAEImageEncoder(nn.Module):
    """输出空间 VAE latent 的均值与对数方差。"""

    def __init__(self, embedding_dim: int, spatial_size: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((spatial_size, spatial_size)),
        )
        self.mean = nn.Conv2d(48, embedding_dim, kernel_size=1)
        self.logvar = nn.Conv2d(48, embedding_dim, kernel_size=1)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(image)
        return self.mean(features), self.logvar(features).clamp(-12.0, 8.0)


class ImageDecoder(nn.Module):
    """从空间 latent 解码成可视化用 RGB 图像。"""

    def __init__(self, embedding_dim: int, image_size: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = embedding_dim
        for _ in range(int(torch.log2(torch.tensor(image_size // 8)).item())):
            next_channels = max(4, channels // 2)
            layers.append(nn.ConvTranspose2d(channels, next_channels, kernel_size=4, stride=2, padding=1))
            layers.append(nn.ReLU())
            channels = next_channels
        layers.append(nn.Conv2d(channels, 3, kernel_size=1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.network(latent)


class SpatialPredictor(nn.Module):
    """把一段动作作为通道条件，预测未来空间 latent。"""

    def __init__(self, embedding_dim: int, action_horizon: int) -> None:
        super().__init__()
        self.action_encoder = nn.Sequential(
            nn.Linear(action_horizon * 2, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
        )
        self.network = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, padding=1),
        )

    def forward(self, context: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        action_features = self.action_encoder(actions.flatten(start_dim=1)).unsqueeze(-1).unsqueeze(-1)
        return context + self.network(context + action_features)


class JEPAModel(nn.Module):
    """当前观察和动作序列到未来目标 embedding 的预测器。"""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.context_encoder = ImageEncoder(self.config.embedding_dim, self.config.spatial_size)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self.target_encoder.requires_grad_(False)
        self.target_encoder.eval()
        self.predictor = SpatialPredictor(self.config.embedding_dim, self.config.action_horizon)
        pose_features = self.config.embedding_dim * self.config.spatial_size * self.config.spatial_size
        self.pose_head = nn.Sequential(nn.Flatten(), nn.Linear(pose_features, 128), nn.ReLU(), nn.Linear(128, 6))
        self.decoder = ImageDecoder(self.config.embedding_dim, self.config.image_size)

    def forward(
        self,
        image: torch.Tensor,
        actions: torch.Tensor,
        future_image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_images(image, "当前图像")
        self._validate_images(future_image, "未来图像")
        self._validate_actions(actions, image.shape[0])
        prediction = self.predict_from_context(self.encode_context(image), actions)
        target = self.encode_target(future_image)
        return prediction, target

    def encode_context(self, image: torch.Tensor) -> torch.Tensor:
        self._validate_images(image, "当前图像")
        return torch.nn.functional.normalize(self.context_encoder(image), dim=1)

    @torch.no_grad()
    def encode_target(self, image: torch.Tensor) -> torch.Tensor:
        self._validate_images(image, "目标图像")
        return torch.nn.functional.normalize(self.target_encoder(image), dim=1)

    @torch.no_grad()
    def encode_goal(self, image: torch.Tensor) -> torch.Tensor:
        """编码只含 T 物体的目标，忽略任务无关的蓝色推杆位置。"""
        return self.encode_target(self._mask_pusher(image))

    @torch.no_grad()
    def encode_goal_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """将预测的完整 latent 解码后，投影到 T 物体目标空间。"""
        return self.encode_goal(self.decode(latent))

    def predict_from_context(self, context: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        expected = (self.config.embedding_dim, self.config.spatial_size, self.config.spatial_size)
        if tuple(context.shape[1:]) != expected:
            raise ValueError("上下文 embedding 形状无效")
        self._validate_actions(actions, context.shape[0])
        return torch.nn.functional.normalize(self.predictor(context, actions), dim=1)

    def decode(self, embedding: torch.Tensor) -> torch.Tensor:
        """把 context、target 或预测 embedding 解码为 RGB 图像。"""
        expected = (self.config.embedding_dim, self.config.spatial_size, self.config.spatial_size)
        if embedding.ndim != 4 or tuple(embedding.shape[1:]) != expected:
            raise ValueError("待解码 embedding 形状无效")
        return self.decoder(embedding)

    def predict_pose(self, embedding: torch.Tensor) -> torch.Tensor:
        """从空间 latent 估计推杆与 T 的几何状态。"""
        expected = (self.config.embedding_dim, self.config.spatial_size, self.config.spatial_size)
        if embedding.ndim != 4 or tuple(embedding.shape[1:]) != expected:
            raise ValueError("待预测位姿的 embedding 形状无效")
        return self.pose_head(embedding)

    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("EMA 动量必须位于 0 到 1 之间")
        for target, context in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            target.mul_(momentum).add_(context, alpha=1.0 - momentum)

    def _validate_images(self, image: torch.Tensor, name: str) -> None:
        if image.ndim != 4 or image.shape[1] != 3 or image.shape[2:] != (self.config.image_size, self.config.image_size):
            raise ValueError(f"{name}图像尺寸与模型配置不一致")

    def _validate_actions(self, actions: torch.Tensor, batch_size: int) -> None:
        expected = (batch_size, self.config.action_horizon, 2)
        if tuple(actions.shape) != expected:
            raise ValueError(f"动作必须是形状为 {expected} 的张量")

    def _mask_pusher(self, image: torch.Tensor) -> torch.Tensor:
        red, green, blue = image.unbind(dim=1)
        pusher = (blue > green + 45 / 255) & (blue > red + 70 / 255)
        background = torch.tensor((245 / 255, 247 / 255, 250 / 255), dtype=image.dtype, device=image.device).view(1, 3, 1, 1)
        return torch.where(pusher.unsqueeze(1), background, image)


class VAEJEPAModel(JEPAModel):
    """以空间 VAE latent 作为 JEPA 世界模型状态的联合模型。"""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__(config)
        self.context_encoder = VAEImageEncoder(self.config.embedding_dim, self.config.spatial_size)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self.target_encoder.requires_grad_(False)
        self.target_encoder.eval()

    def encode_distribution(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_images(image, "当前图像")
        return self.context_encoder(image)

    def sample_latent(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            return mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        return mean

    def kl_loss(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return 0.5 * (mean.square() + logvar.exp() - 1.0 - logvar).mean()

    def encode_context(self, image: torch.Tensor) -> torch.Tensor:
        mean, logvar = self.encode_distribution(image)
        return torch.nn.functional.normalize(self.sample_latent(mean, logvar), dim=1)

    @torch.no_grad()
    def encode_target(self, image: torch.Tensor) -> torch.Tensor:
        self._validate_images(image, "目标图像")
        mean, _ = self.target_encoder(image)
        return torch.nn.functional.normalize(mean, dim=1)

    def forward(self, image: torch.Tensor, actions: torch.Tensor, future_image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_images(future_image, "未来图像")
        self._validate_actions(actions, image.shape[0])
        mean, logvar = self.encode_distribution(image)
        context = torch.nn.functional.normalize(self.sample_latent(mean, logvar), dim=1)
        return self.predict_from_context(context, actions), self.encode_target(future_image), mean, logvar


def model_from_checkpoint_config(config: Mapping[str, object]) -> JEPAModel:
    """根据检查点配置构造相应模型；缺失类型字段的旧检查点视为普通 JEPA。"""
    model_type = config.get("model_type", "jepa")
    if model_type not in {"jepa", "vae_jepa"}:
        raise ValueError(f"不支持的检查点模型类型: {model_type}")
    model_config = ModelConfig(
        image_size=int(config.get("image_size", 64)),
        action_horizon=int(config.get("action_horizon", 4)),
    )
    return VAEJEPAModel(model_config) if model_type == "vae_jepa" else JEPAModel(model_config)

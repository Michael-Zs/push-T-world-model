"""动作条件 JEPA：只预测未来表征，不重建像素。"""

from __future__ import annotations

import copy

import torch
from torch import nn

from .config import ModelConfig


class ImageEncoder(nn.Module):
    """用于 64x64 RGB 观察的小型卷积编码器。"""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(48, embedding_dim),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.network(image)


class ImageDecoder(nn.Module):
    """将 embedding 解码成可视化用的 64x64 RGB 图像。"""

    def __init__(self, embedding_dim: int, image_size: int) -> None:
        super().__init__()
        self.project = nn.Linear(embedding_dim, 48 * 8 * 8)
        layers: list[nn.Module] = []
        channels = 48
        for _ in range(int(torch.log2(torch.tensor(image_size // 8)).item())):
            next_channels = max(4, channels // 2)
            layers.append(nn.ConvTranspose2d(channels, next_channels, kernel_size=4, stride=2, padding=1))
            layers.append(nn.ReLU())
            channels = next_channels
        layers.append(nn.Conv2d(channels, 3, kernel_size=1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        features = self.project(embedding).reshape(-1, 48, 8, 8)
        return self.network(features)


class JEPAModel(nn.Module):
    """当前观察和动作序列到未来目标 embedding 的预测器。"""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.context_encoder = ImageEncoder(self.config.embedding_dim)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self.target_encoder.requires_grad_(False)
        self.target_encoder.eval()
        self.predictor = nn.Sequential(
            nn.Linear(self.config.embedding_dim + self.config.action_horizon * 2, 128),
            nn.ReLU(),
            nn.Linear(128, self.config.embedding_dim),
        )
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

    def predict_from_context(self, context: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if context.ndim != 2 or context.shape[1] != self.config.embedding_dim:
            raise ValueError("上下文 embedding 形状无效")
        self._validate_actions(actions, context.shape[0])
        return torch.nn.functional.normalize(self.predictor(torch.cat((context, actions.flatten(start_dim=1)), dim=1)), dim=1)

    def decode(self, embedding: torch.Tensor) -> torch.Tensor:
        """把 context、target 或预测 embedding 解码为 RGB 图像。"""
        if embedding.ndim != 2 or embedding.shape[1] != self.config.embedding_dim:
            raise ValueError("待解码 embedding 形状无效")
        return self.decoder(embedding)

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

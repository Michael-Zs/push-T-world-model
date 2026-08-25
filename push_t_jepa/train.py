"""JEPA 的 CPU 训练、验证和检查点工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch.nn import functional as functional

from .model import JEPAModel


def train_epoch(
    model: JEPAModel,
    loader: Iterable[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    ema_momentum: float,
) -> float:
    """运行一个 epoch，并返回平均 embedding MSE。"""
    model.train()
    losses: list[float] = []
    for batch in loader:
        image, actions, future_image = _batch_to_device(batch, _model_device(model))
        optimizer.zero_grad()
        prediction, target = model(image, actions, future_image)
        loss = functional.mse_loss(prediction, target)
        loss.backward()
        optimizer.step()
        model.update_target_encoder(ema_momentum)
        losses.append(float(loss.detach().cpu()))
    if not losses:
        raise ValueError("训练数据不能为空")
    return sum(losses) / len(losses)


@torch.no_grad()
def validate(model: JEPAModel, loader: Iterable[dict[str, torch.Tensor]]) -> float:
    """计算验证集上的平均 embedding MSE。"""
    model.eval()
    losses: list[float] = []
    for batch in loader:
        image, actions, future_image = _batch_to_device(batch, _model_device(model))
        prediction, target = model(image, actions, future_image)
        losses.append(float(functional.mse_loss(prediction, target).cpu()))
    if not losses:
        raise ValueError("验证数据不能为空")
    return sum(losses) / len(losses)


def save_checkpoint(
    path: str | Path,
    model: JEPAModel,
    config: dict[str, object],
    metrics: dict[str, float],
) -> None:
    """保存模型、配置和可序列化训练指标。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": config, "metrics": metrics}, target)


def load_checkpoint(path: str | Path, model: JEPAModel, device: str = "cpu") -> dict[str, object]:
    """加载检查点并返回其中保存的配置和指标。"""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"检查点不存在: {source}")
    checkpoint = torch.load(source, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError("检查点缺少模型参数")
    model.load_state_dict(checkpoint["model_state"])
    return {"config": checkpoint.get("config", {}), "metrics": checkpoint.get("metrics", {})}


def _model_device(model: JEPAModel) -> torch.device:
    return next(model.parameters()).device


def _batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    required = ("image", "actions", "future_image")
    if any(key not in batch for key in required):
        raise ValueError("训练批次缺少图像、动作或未来图像")
    return tuple(batch[key].to(device) for key in required)  # type: ignore[return-value]

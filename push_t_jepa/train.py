"""JEPA 的 CPU 训练、验证和检查点工具。"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Callable, Iterable

import torch
from torch.nn import functional as functional
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from .config import EnvConfig, ModelConfig
from .dataset import PushTJEPADataset, collect_trajectories, collect_trajectories_with_stats
from .model import JEPAModel


def train_epoch(
    model: JEPAModel,
    loader: Iterable[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    ema_momentum: float,
    variance_weight: float = 0.1,
    reconstruction_weight: float = 0.25,
    progress: Callable[[int, int, float], None] | None = None,
) -> float:
    """运行一个 epoch，并返回平均 embedding MSE。"""
    model.train()
    losses: list[float] = []
    total_batches = len(loader) if hasattr(loader, "__len__") else 0
    for batch_index, batch in enumerate(loader, start=1):
        image, actions, future_image = _batch_to_device(batch, _model_device(model))
        optimizer.zero_grad()
        prediction, target = model(image, actions, future_image)
        embedding_loss, _ = jepa_loss(prediction, target, variance_weight)
        pose_loss = torch.zeros((), device=image.device)
        if "state" in batch and "future_state" in batch:
            state = batch["state"].to(_model_device(model))
            future_state = batch["future_state"].to(_model_device(model))
            pose_loss = functional.mse_loss(model.predict_pose(model.encode_context(image)), state)
            pose_loss = pose_loss + functional.mse_loss(model.predict_pose(prediction), future_state)
        # decoder 仅用于可视化；阻断其梯度，避免像素重建牺牲 JEPA 动力学表征。
        reconstruction_loss = _foreground_reconstruction_loss(model.decode(model.encode_context(image).detach()), image)
        prediction_reconstruction_loss = _foreground_reconstruction_loss(model.decode(prediction.detach()), future_image)
        loss = embedding_loss + pose_loss + reconstruction_weight * (reconstruction_loss + prediction_reconstruction_loss)
        loss.backward()
        optimizer.step()
        model.update_target_encoder(ema_momentum)
        losses.append(float(loss.detach().cpu()))
        if progress is not None:
            progress(batch_index, total_batches, losses[-1])
    if not losses:
        raise ValueError("训练数据不能为空")
    return sum(losses) / len(losses)


def jepa_loss(prediction: torch.Tensor, target: torch.Tensor, variance_weight: float = 0.1) -> tuple[torch.Tensor, float]:
    """返回预测 MSE 加方差下界正则，并报告预测 embedding 平均标准差。"""
    if variance_weight < 0.0:
        raise ValueError("方差正则权重不能为负数")
    mse = functional.mse_loss(prediction, target)
    samples = prediction.movedim(1, -1).reshape(-1, prediction.shape[1])
    std = torch.sqrt(samples.var(dim=0, unbiased=False) + 1e-4)
    variance_penalty = torch.relu(1.0 - std).mean()
    return mse + variance_weight * variance_penalty, float(std.mean().detach().cpu())


def _foreground_reconstruction_loss(prediction: torch.Tensor, target: torch.Tensor, foreground_weight: float = 12.0) -> torch.Tensor:
    """对 Push-T 前景加权，避免 RGB MSE 被大面积浅色背景主导。"""
    if prediction.shape != target.shape:
        raise ValueError("重建预测与目标图像形状必须一致")
    background = target.new_tensor((245 / 255, 247 / 255, 250 / 255)).view(1, 3, 1, 1)
    foreground = (target - background).abs().amax(dim=1, keepdim=True) > 0.08
    weights = 1.0 + foreground.to(target.dtype) * (foreground_weight - 1.0)
    squared_error = (prediction - target).square()
    return (squared_error * weights).sum() / (weights.sum() * prediction.shape[1])


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


@torch.no_grad()
def validate_pose(model: JEPAModel, loader: Iterable[dict[str, torch.Tensor]]) -> float:
    """计算未来 T 物体位姿（位置、sin/cos 朝向）的预测 MSE。"""
    model.eval()
    losses: list[float] = []
    for batch in loader:
        if "future_state" not in batch:
            continue
        image, actions, future_image = _batch_to_device(batch, _model_device(model))
        prediction, _ = model(image, actions, future_image)
        target_pose = batch["future_state"].to(_model_device(model))[:, 2:]
        losses.append(float(functional.mse_loss(model.predict_pose(prediction)[:, 2:], target_pose).cpu()))
    if not losses:
        return 0.0
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
    incompatibility = model.load_state_dict(checkpoint["model_state"], strict=False)
    missing_non_decoder = [key for key in incompatibility.missing_keys if not key.startswith("decoder.")]
    if missing_non_decoder or incompatibility.unexpected_keys:
        raise ValueError("检查点模型结构不兼容，缺少或包含未知参数")
    decoder_available = not incompatibility.missing_keys
    if not decoder_available:
        warnings.warn("检查点不包含 decoder 参数：CEM 规划可继续，但 GIF 的预测解码栏未经训练；请重新训练以获得有效图像。", stacklevel=2)
    return {"config": checkpoint.get("config", {}), "metrics": checkpoint.get("metrics", {}), "decoder_available": decoder_available}


def _model_device(model: JEPAModel) -> torch.device:
    return next(model.parameters()).device


def _batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    required = ("image", "actions", "future_image")
    if any(key not in batch for key in required):
        raise ValueError("训练批次缺少图像、动作或未来图像")
    return tuple(batch[key].to(device) for key in required)  # type: ignore[return-value]


def run_smoke_training(output: str | Path, seed: int = 7) -> Path:
    """以极小数据集完成一次 CPU 训练，供安装验证和演示使用。"""
    return run_training(output, trajectories=8, steps=8, epochs=1, batch_size=8, seed=seed)


def run_training(
    output: str | Path,
    trajectories: int = 3_000,
    steps: int = 24,
    epochs: int = 60,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    variance_weight: float = 0.1,
    reconstruction_weight: float = 0.25,
    image_size: int = 64,
    action_horizon: int = 4,
    threads: int | None = None,
    seed: int = 7,
    progress: Callable[[str], None] | None = None,
    resume_checkpoint: str | Path | None = None,
) -> Path:
    """训练并保存验证损失最优的 CPU JEPA 检查点。"""
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("训练轮数和批量大小必须为正数")
    worker_threads = threads if threads is not None else (os.cpu_count() or 1)
    if worker_threads <= 0:
        raise ValueError("训练线程数必须为正数")
    torch.set_num_threads(worker_threads)
    torch.manual_seed(seed)
    env_config = EnvConfig(image_size=image_size)
    samples, collection_stats = collect_trajectories_with_stats(env_config=env_config, trajectories=trajectories, steps=steps, seed=seed, progress=(lambda done, total: progress(f"采集轨迹: {done}/{total}") if progress is not None and (done == total or done % max(1, total // 20) == 0) else None))
    dataset = PushTJEPADataset(samples, horizon=action_horizon)
    validation_size = max(1, len(dataset) // 10)
    train_size = len(dataset) - validation_size
    if train_size <= 0:
        raise ValueError("数据量不足以划分训练集和验证集")
    train_set, validation_set = random_split(dataset, [train_size, validation_size], generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
    model = JEPAModel(ModelConfig(image_size=image_size, action_horizon=action_horizon))
    if resume_checkpoint is not None:
        load_checkpoint(resume_checkpoint, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    result = Path(output)
    result.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=result / "tensorboard")
    writer.add_scalar("data/effective_step_rate", collection_stats.effective_step_rate, 0)
    writer.add_scalar("data/mean_translation", collection_stats.mean_translation, 0)
    writer.add_scalar("data/mean_rotation", collection_stats.mean_rotation, 0)
    history: list[dict[str, float]] = []
    best_validation = float("inf")
    checkpoint = result / "model.pt"
    config = {"seed": seed, "trajectories": trajectories, "steps": steps, "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate, "variance_weight": variance_weight, "reconstruction_weight": reconstruction_weight, "image_size": image_size, "action_horizon": action_horizon, "threads": worker_threads}
    for epoch in range(1, epochs + 1):
        def batch_progress(current: int, total: int, loss: float) -> None:
            global_step = (epoch - 1) * total + current
            writer.add_scalar("train/batch_loss", loss, global_step)
            if progress is not None and (current == total or current % max(1, total // 10) == 0):
                progress(f"Epoch {epoch}/{epochs} 批次 {current}/{total} loss={loss:.5f}")
        train_loss = train_epoch(model, train_loader, optimizer, 0.99, variance_weight, reconstruction_weight, batch_progress)
        validation_loss = validate(model, validation_loader)
        pose_validation = validate_pose(model, validation_loader)
        writer.add_scalar("train/epoch_loss", train_loss, epoch)
        writer.add_scalar("validation/mse", validation_loss, epoch)
        writer.add_scalar("validation/t_pose_mse", pose_validation, epoch)
        selection_loss = validation_loss + pose_validation
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_mse": validation_loss, "validation_t_pose_mse": pose_validation})
        if selection_loss < best_validation:
            best_validation = selection_loss
            save_checkpoint(checkpoint, model, config, {"best_validation_score": best_validation, "validation_mse": validation_loss, "validation_t_pose_mse": pose_validation, "epoch": epoch})
        writer.add_scalar("validation/best_score", best_validation, epoch)
        preview = next(iter(validation_loader))
        with torch.no_grad():
            image = preview["image"][:4]
            actions = preview["actions"][:4]
            future = preview["future_image"][:4]
            prediction, _ = model(image, actions, future)
            writer.add_images("images/current", image, epoch)
            writer.add_images("images/future_target", future, epoch)
            writer.add_images("images/predicted_decoder", model.decode(prediction), epoch)
        if progress is not None:
            progress(f"Epoch {epoch}/{epochs} 完成 train={train_loss:.5f} val={validation_loss:.5f} best={best_validation:.5f}")
    (result / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    writer.close()
    return checkpoint


def main() -> None:
    """提供 `python -m push_t_jepa.train --smoke` 命令。"""
    import argparse

    parser = argparse.ArgumentParser(description="训练 CPU 优先的 Push-T JEPA 模型")
    parser.add_argument("--smoke", action="store_true", help="运行极小数据集的 smoke 训练")
    parser.add_argument("--output", default="artifacts/train", help="检查点输出目录")
    parser.add_argument("--seed", type=int, default=7, help="随机种子")
    parser.add_argument("--trajectories", type=int, default=3_000, help="采集轨迹数量")
    parser.add_argument("--steps", type=int, default=24, help="每条轨迹步数")
    parser.add_argument("--epochs", type=int, default=60, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=64, help="批量大小")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Adam 学习率")
    parser.add_argument("--variance-weight", type=float, default=0.1, help="反坍塌方差正则权重")
    parser.add_argument("--reconstruction-weight", type=float, default=0.25, help="图像解码重建损失权重")
    parser.add_argument("--image-size", type=int, default=64, help="环境、模型与 decoder 共用的正方形图像尺寸")
    parser.add_argument("--action-horizon", type=int, default=4, help="JEPA 直接预测的动作步数；设为 8 可避免 CEM 两段递推")
    parser.add_argument("--threads", type=int, default=None, help="PyTorch CPU 计算线程数，默认使用所有逻辑核")
    parser.add_argument("--resume", default=None, help="从兼容的检查点继续训练（优化器状态会重新初始化）")
    args = parser.parse_args()
    checkpoint = run_smoke_training(args.output, seed=args.seed) if args.smoke else run_training(
        args.output, args.trajectories, args.steps, args.epochs, args.batch_size,
        args.learning_rate, args.variance_weight, args.reconstruction_weight, args.image_size, args.action_horizon, args.threads, args.seed, print, args.resume,
    )
    print(f"训练完成，检查点位于: {checkpoint}")


if __name__ == "__main__":
    main()

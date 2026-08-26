"""加载 JEPA 模型并导出 CEM 滚动规划的 GIF 演示。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .config import EnvConfig, ModelConfig
from .env import PushTEnv
from .model import JEPAModel
from .planner import CEMPlanner
from .train import load_checkpoint


def run_demo(checkpoint: str | Path, output: str | Path, seed: int = 7, steps: int = 12) -> Path:
    """以固定目标位姿进行滚动 CEM 规划，并写出 GIF 和指标 JSON。"""
    if steps < 0:
        raise ValueError("演示步数不能为负数")
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_config = checkpoint_data.get("config", {}) if isinstance(checkpoint_data, dict) else {}
    image_size = int(checkpoint_config.get("image_size", 64))
    model = JEPAModel(ModelConfig(image_size=image_size))
    load_checkpoint(checkpoint, model)
    env = PushTEnv(config=EnvConfig(image_size=image_size), seed=seed)
    current = env.reset()
    target_env = PushTEnv(config=EnvConfig(image_size=image_size), seed=seed + 1)
    target_env.reset()
    target_env.set_state(
        pusher=np.array([0.15, 0.15], dtype=np.float32),
        object_position=np.array([0.72, 0.62], dtype=np.float32),
        object_angle=0.45,
    )
    target_image = target_env.render()
    target_position = target_env.state.object_position.copy()
    initial_distance = float(np.linalg.norm(env.state.object_position - target_position))
    real_frames = [current]
    frames: list[np.ndarray] = []
    planner = CEMPlanner(model, seed=seed)
    with torch.no_grad():
        target_embedding = model.encode_target(planner._image_tensor(target_image, torch.device("cpu")))
        initial_embedding = model.encode_context(planner._image_tensor(current, torch.device("cpu")))
    for _ in range(steps):
        approach_action = planner.approach_action(real_frames[-1])
        sequence = planner.plan(real_frames[-1], target_image) if approach_action is None else np.zeros((planner.config.horizon, 2), dtype=np.float32)
        if approach_action is not None:
            sequence[0] = approach_action
        with torch.no_grad():
            context = model.encode_context(planner._image_tensor(real_frames[-1], torch.device("cpu")))
            predicted_embedding = model.predict_from_context(
                context,
                torch.from_numpy(sequence[: model.config.action_horizon]).unsqueeze(0),
            )
            predicted_image = _decoder_image(model.decode(predicted_embedding))
        action = planner.next_action(real_frames[-1], target_image)
        frame, _ = env.step(action)
        real_frames.append(frame)
        frames.append(_compose_frame(frame, predicted_image, target_image))
    result = Path(output)
    result.mkdir(parents=True, exist_ok=True)
    if not frames:
        frames.append(_compose_frame(current, current, target_image))
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(result / "rollout.gif", save_all=True, append_images=images[1:], duration=120, loop=0)
    with torch.no_grad():
        final_embedding = model.encode_context(planner._image_tensor(real_frames[-1], torch.device("cpu")))
    metrics = {
        "seed": seed,
        "steps": steps,
        "initial_geometric_distance": initial_distance,
        "final_geometric_distance": float(np.linalg.norm(env.state.object_position - target_position)),
        "initial_embedding_distance": float((initial_embedding - target_embedding).square().sum().sqrt()),
        "final_embedding_distance": float((final_embedding - target_embedding).square().sum().sqrt()),
    }
    (result / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _decoder_image(decoded: torch.Tensor) -> np.ndarray:
    return decoded[0].detach().cpu().permute(1, 2, 0).mul(255).clamp(0, 255).to(torch.uint8).numpy()


def _compose_frame(real: np.ndarray, predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """拼接真实执行、JEPA 解码预测和目标图像，便于并排观察。"""
    return np.concatenate((real, predicted, target), axis=1)


def main() -> None:
    """提供 `python -m push_t_jepa.demo` 命令。"""
    import argparse

    parser = argparse.ArgumentParser(description="运行 Push-T JEPA 的 CEM 规划演示")
    parser.add_argument("--checkpoint", required=True, help="训练检查点路径")
    parser.add_argument("--output", default="artifacts/demo", help="演示输出目录")
    parser.add_argument("--seed", type=int, default=7, help="随机种子")
    parser.add_argument("--steps", type=int, default=12, help="滚动执行步数")
    args = parser.parse_args()
    output = run_demo(args.checkpoint, args.output, seed=args.seed, steps=args.steps)
    print(f"演示已写入: {output}")


if __name__ == "__main__":
    main()

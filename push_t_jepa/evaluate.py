"""以多个随机初始状态评估 JEPA 规划，并和随机动作基线比较。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import EnvConfig
from .demo import run_demo
from .env import PushTEnv


def summarize_distances(distances: Iterable[tuple[float, float]]) -> dict[str, float | int]:
    """汇总若干组 ``(初始距离, 最终距离)``。"""
    values = list(distances)
    if not values:
        raise ValueError("至少需要一组距离")
    initial = float(np.mean([pair[0] for pair in values]))
    final = float(np.mean([pair[1] for pair in values]))
    return {"count": len(values), "mean_initial_distance": initial, "mean_final_distance": final, "mean_improvement": initial - final}


def run_evaluation(checkpoint: str | Path, output: str | Path, seeds: Iterable[int], steps: int = 80) -> Path:
    """运行规划和随机动作，并写出逐 seed 与聚合指标。"""
    if steps <= 0:
        raise ValueError("执行步数必须为正数")
    result = Path(output)
    result.mkdir(parents=True, exist_ok=True)
    planner_rows: list[dict[str, float | int]] = []
    random_rows: list[dict[str, float | int]] = []
    for seed in seeds:
        demo_path = run_demo(checkpoint, result / f"seed-{seed}", seed=seed, steps=steps)
        metrics = json.loads((demo_path / "metrics.json").read_text(encoding="utf-8"))
        planner_rows.append(metrics)
        random_rows.append(_random_baseline(seed, steps))
    payload = {
        "steps": steps,
        "planner": {"runs": planner_rows, "summary": summarize_distances([(float(row["initial_geometric_distance"]), float(row["final_geometric_distance"])) for row in planner_rows])},
        "random": {"runs": random_rows, "summary": summarize_distances([(float(row["initial_geometric_distance"]), float(row["final_geometric_distance"])) for row in random_rows])},
    }
    path = result / "evaluation.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _random_baseline(seed: int, steps: int) -> dict[str, float | int]:
    env = PushTEnv(config=EnvConfig(), seed=seed)
    env.reset()
    target = np.array([0.72, 0.62], dtype=np.float32)
    initial = float(np.linalg.norm(env.state.object_position - target))
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        env.step(rng.uniform(-1.0, 1.0, size=2).astype(np.float32))
    return {"seed": seed, "initial_geometric_distance": initial, "final_geometric_distance": float(np.linalg.norm(env.state.object_position - target))}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="多 seed 评估 Push-T JEPA 规划与随机动作基线")
    parser.add_argument("--checkpoint", required=True, help="模型检查点路径")
    parser.add_argument("--output", default="artifacts/evaluation", help="评估产物目录")
    parser.add_argument("--seeds", default="3,5,7,9,11", help="以逗号分隔的随机种子")
    parser.add_argument("--steps", type=int, default=80, help="每个 seed 的执行步数")
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    path = run_evaluation(args.checkpoint, args.output, seeds, args.steps)
    print(f"评估已写入: {path}")


if __name__ == "__main__":
    main()

# Push-T JEPA 与 CEM 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 CPU 上实现可训练、可测试的手写 Push-T 仿真、动作条件 JEPA 世界模型和基于目标 embedding 的 CEM 规划演示。

**Architecture:** `PushTEnv` 产生确定性 RGB 观察和动作序列轨迹；JEPA 以当前观察与未来动作预测未来观察的 EMA 目标 embedding。CEM 在动作序列分布上迭代采样，最小化预测终点 embedding 与目标 embedding 的距离，并使用滚动时域执行。

**Tech Stack:** Python 3.10+、NumPy、PyTorch、Pillow、pytest。

**Spec:** `docs/superpowers/specs/2026-08-26-push-t-jepa-design.md`

## Global Constraints

- 所有源代码、命令行帮助与错误信息使用中文。
- 默认使用 CPU，且不得依赖 Gymnasium、Pymunk、Pygame、CUDA 或在线服务。
- 默认图像为 `64x64x3`，embedding 为 64 维，JEPA 预测跨度为 4 步。
- 所有随机过程都接受显式种子；相同种子必须产生相同结果。
- `artifacts/` 必须在 `.gitignore` 中，训练产物不可提交。
- 每个行为变更均先写 pytest 失败测试，再实现最小代码；保留失败与通过的命令输出作为执行证据。

---

## 文件结构

| 文件 | 职责 |
| --- | --- |
| `pyproject.toml` | 项目元数据、运行时依赖、pytest 配置和命令入口。 |
| `.gitignore` | 排除虚拟环境、缓存、数据和 `artifacts/`。 |
| `push_t_jepa/config.py` | 冻结的环境、训练和 CEM 配置数据类。 |
| `push_t_jepa/env.py` | 状态、近似推动动力学、动作校验与 Pillow 渲染。 |
| `push_t_jepa/dataset.py` | 确定性轨迹采集和 `(image, actions, future_image)` 数据集。 |
| `push_t_jepa/model.py` | CNN 编码器、动作预测器和 EMA 目标编码器的 JEPA 模型。 |
| `push_t_jepa/planner.py` | CEM 采样、代价计算和滚动时域控制。 |
| `push_t_jepa/train.py` | 训练/验证循环、检查点和指标写入。 |
| `push_t_jepa/demo.py` | 加载模型、构造目标、规划、导出 GIF 和 JSON 指标。 |
| `tests/*.py` | 上述模块与端到端 smoke 测试。 |
| `README.md` | 安装、测试、训练和演示命令及原理说明。 |

## 任务 1：项目骨架和配置

**文件：**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `push_t_jepa/__init__.py`
- Create: `push_t_jepa/config.py`
- Test: `tests/test_config.py`

**接口：**
- 产生：`EnvConfig(image_size: int = 64, arena_size: float = 1.0, max_action: float = 0.08)`；`ModelConfig(embedding_dim: int = 64, action_horizon: int = 4)`；`CEMConfig(horizon: int = 8, population: int = 128, elite_count: int = 16, iterations: int = 3)`；`TrainConfig(seed: int = 7, device: str = "cpu")`。
- 约束：各配置数据类是 `@dataclass(frozen=True)`；`CEMConfig` 对非正参数和 `elite_count > population` 抛出中文 `ValueError`。

- [ ] **步骤 1：编写失败测试**

```python
from push_t_jepa.config import CEMConfig, EnvConfig, ModelConfig
import pytest

def test_default_cpu_configuration_has_design_values():
    assert EnvConfig().image_size == 64
    assert ModelConfig().embedding_dim == 64
    assert CEMConfig().population == 128

def test_cem_rejects_more_elites_than_candidates():
    with pytest.raises(ValueError, match="精英"):
        CEMConfig(population=4, elite_count=5)
```

- [ ] **步骤 2：运行测试，确认失败原因是模块尚不存在**

Run: `pytest -q tests/test_config.py`

Expected: FAIL，显示 `ModuleNotFoundError: No module named 'push_t_jepa'`。

- [ ] **步骤 3：实现最小骨架和配置**

```python
@dataclass(frozen=True)
class CEMConfig:
    horizon: int = 8
    population: int = 128
    elite_count: int = 16
    iterations: int = 3

    def __post_init__(self) -> None:
        if self.elite_count > self.population:
            raise ValueError("精英数量不能大于候选数量")
```

在 `pyproject.toml` 声明 `numpy`、`torch`、`pillow` 和 `pytest`，并以 src 同级包方式让 pytest 发现 `push_t_jepa`。

- [ ] **步骤 4：运行测试，确认通过**

Run: `pytest -q tests/test_config.py`

Expected: PASS，2 项通过。

- [ ] **步骤 5：提交**

```bash
git add pyproject.toml .gitignore push_t_jepa tests/test_config.py
git commit -m "chore: scaffold Push-T JEPA project"
```

## 任务 2：确定性 Push-T 环境和渲染

**文件：**
- Create: `push_t_jepa/env.py`
- Test: `tests/test_env.py`

**接口：**
- 消费：`EnvConfig`。
- 产生：`PushTState(pusher: np.ndarray, object_position: np.ndarray, object_angle: float)`；`PushTEnv(config: EnvConfig, seed: int)`；`reset(seed: int | None = None) -> np.ndarray`；`step(action: np.ndarray) -> tuple[np.ndarray, PushTState]`；`render() -> np.ndarray`；`state -> PushTState`。
- `render()` 返回 `np.uint8` 的 `(64, 64, 3)` RGB 数组；`step` 仅接受形状 `(2,)` 且有限的动作，非法输入抛出中文 `ValueError`。

- [ ] **步骤 1：编写失败测试**

```python
import numpy as np
from push_t_jepa.env import PushTEnv

def test_reset_is_seed_deterministic_and_returns_rgb_image():
    first = PushTEnv(seed=3).reset()
    second = PushTEnv(seed=3).reset()
    assert first.shape == (64, 64, 3)
    assert first.dtype == np.uint8
    assert np.array_equal(first, second)

def test_contact_push_changes_t_object_pose():
    env = PushTEnv(seed=0)
    env.reset()
    env.set_state(pusher=np.array([0.42, 0.50]), object_position=np.array([0.50, 0.50]), object_angle=0.0)
    before = env.state.object_position.copy()
    env.step(np.array([1.0, 0.0]))
    assert env.state.object_position[0] > before[0]
```

- [ ] **步骤 2：运行测试，确认失败**

Run: `pytest -q tests/test_env.py`

Expected: FAIL，显示 `push_t_jepa.env` 不存在。

- [ ] **步骤 3：实现最小环境**

实现 `set_state` 仅作为确定性测试辅助接口；将归一化动作乘 `max_action` 后裁剪推杆位置。以 T 的局部矩形联合边界盒进行接触检测；接触时按推动方向平移并按照接触点相对物体中心的二维叉积更新角度。使用 `PIL.Image` 与 `ImageDraw.polygon` 绘制深色 T、蓝色推杆和浅色背景。

- [ ] **步骤 4：补充边界和非法动作测试，然后运行**

```python
import pytest

def test_step_clamps_pusher_inside_arena():
    env = PushTEnv(seed=0)
    env.reset()
    env.set_state(pusher=np.array([0.99, 0.99]), object_position=np.array([0.5, 0.5]), object_angle=0.0)
    env.step(np.array([1.0, 1.0]))
    assert np.all(env.state.pusher <= 1.0)

def test_step_rejects_wrong_action_shape():
    with pytest.raises(ValueError, match="动作"):
        PushTEnv(seed=0).step(np.array([0.0]))
```

Run: `pytest -q tests/test_env.py`

Expected: PASS，4 项通过。

- [ ] **步骤 5：提交**

```bash
git add push_t_jepa/env.py tests/test_env.py
git commit -m "feat: add deterministic Push-T environment"
```

## 任务 3：轨迹采集与数据集

**文件：**
- Create: `push_t_jepa/dataset.py`
- Test: `tests/test_dataset.py`

**接口：**
- 消费：`PushTEnv`、`ModelConfig.action_horizon`。
- 产生：`collect_trajectories(env_config: EnvConfig, trajectories: int, steps: int, seed: int) -> list[Trajectory]`；`PushTJEPADataset(trajectories: list[Trajectory], horizon: int)`；`__getitem__(index) -> dict[str, torch.Tensor]`。
- 每个样本字典精确包含 `image: float32[3,64,64]`、`actions: float32[horizon,2]` 与 `future_image: float32[3,64,64]`；图像值在 `[0, 1]`。

- [ ] **步骤 1：编写失败测试**

```python
import torch
from push_t_jepa.dataset import PushTJEPADataset, collect_trajectories

def test_collected_trajectories_are_seed_deterministic():
    a = collect_trajectories(trajectories=2, steps=7, seed=11)
    b = collect_trajectories(trajectories=2, steps=7, seed=11)
    assert torch.equal(a[0].actions, b[0].actions)

def test_dataset_returns_normalized_action_conditioned_sample():
    trajectories = collect_trajectories(trajectories=1, steps=8, seed=2)
    sample = PushTJEPADataset(trajectories, horizon=4)[0]
    assert sample["image"].shape == (3, 64, 64)
    assert sample["actions"].shape == (4, 2)
    assert sample["future_image"].max() <= 1.0
```

- [ ] **步骤 2：运行测试，确认失败**

Run: `pytest -q tests/test_dataset.py`

Expected: FAIL，显示 `push_t_jepa.dataset` 不存在。

- [ ] **步骤 3：实现采集和索引映射**

使用局部 `np.random.Generator(seed)` 生成随机动作并记录每个时刻的渲染图像；建立只包含 `t + horizon < trajectory_length` 的样本索引。采集函数拒绝非正轨迹数、过短轨迹和非正步数并报中文 `ValueError`。

- [ ] **步骤 4：运行测试，确认通过**

Run: `pytest -q tests/test_dataset.py`

Expected: PASS，2 项通过。

- [ ] **步骤 5：提交**

```bash
git add push_t_jepa/dataset.py tests/test_dataset.py
git commit -m "feat: add Push-T trajectory dataset"
```

## 任务 4：动作条件 JEPA 模型与 EMA

**文件：**
- Create: `push_t_jepa/model.py`
- Test: `tests/test_model.py`

**接口：**
- 消费：`ModelConfig`、批量 `image[B,3,64,64]`、`actions[B,4,2]`、`future_image[B,3,64,64]`。
- 产生：`JEPAModel(config: ModelConfig)`；`forward(image, actions, future_image) -> tuple[torch.Tensor, torch.Tensor]`；`encode_context(image) -> torch.Tensor`；`encode_target(image) -> torch.Tensor`；`update_target_encoder(momentum: float) -> None`。
- `forward` 返回 `(prediction[B,64], target[B,64])`；目标无梯度；非法输入形状抛出中文 `ValueError`。

- [ ] **步骤 1：编写失败测试**

```python
import torch
from push_t_jepa.model import JEPAModel

def test_jepa_predicts_embedding_and_stops_target_gradient():
    model = JEPAModel()
    prediction, target = model(torch.rand(2, 3, 64, 64), torch.rand(2, 4, 2), torch.rand(2, 3, 64, 64))
    assert prediction.shape == (2, 64)
    assert target.shape == (2, 64)
    assert target.requires_grad is False

def test_ema_moves_target_parameters_after_context_change():
    model = JEPAModel()
    before = next(model.target_encoder.parameters()).detach().clone()
    with torch.no_grad():
        next(model.context_encoder.parameters()).add_(1.0)
    model.update_target_encoder(momentum=0.5)
    assert not torch.equal(before, next(model.target_encoder.parameters()))
```

- [ ] **步骤 2：运行测试，确认失败**

Run: `pytest -q tests/test_model.py`

Expected: FAIL，显示 `push_t_jepa.model` 不存在。

- [ ] **步骤 3：实现小型 CNN、预测器和 EMA**

编码器使用三个带 ReLU 的卷积块、全局平均池化和线性投影；预测器展平动作并与上下文 embedding 拼接，再经两层 MLP 输出 64 维预测。构造时复制上下文编码器参数至目标编码器，并将目标编码器设为 `eval()` 和 `requires_grad_(False)`；EMA 更新用 `target = momentum * target + (1 - momentum) * context`。

- [ ] **步骤 4：补充形状校验测试并运行**

```python
import pytest

def test_jepa_rejects_wrong_action_horizon():
    with pytest.raises(ValueError, match="动作"):
        JEPAModel()(torch.rand(1, 3, 64, 64), torch.rand(1, 3, 2), torch.rand(1, 3, 64, 64))
```

Run: `pytest -q tests/test_model.py`

Expected: PASS，3 项通过。

- [ ] **步骤 5：提交**

```bash
git add push_t_jepa/model.py tests/test_model.py
git commit -m "feat: add action conditioned JEPA model"
```

## 任务 5：训练、验证和检查点

**文件：**
- Create: `push_t_jepa/train.py`
- Test: `tests/test_train.py`

**接口：**
- 消费：`PushTJEPADataset`、`JEPAModel`、`TrainConfig`。
- 产生：`train_epoch(model, loader, optimizer, ema_momentum) -> float`；`validate(model, loader) -> float`；`save_checkpoint(path, model, config, metrics) -> None`；`load_checkpoint(path, model, device="cpu") -> dict[str, object]`。
- 损失为预测和目标 embedding 的 MSE；`train_epoch` 每个优化步骤后更新 EMA；保存路径父目录自动创建。

- [ ] **步骤 1：编写失败测试**

```python
import math
import torch
from torch.utils.data import DataLoader, TensorDataset
from push_t_jepa.model import JEPAModel
from push_t_jepa.train import train_epoch, save_checkpoint, load_checkpoint

def test_one_training_epoch_returns_finite_loss_and_checkpoint_round_trips(tmp_path):
    batch = {"image": torch.rand(4, 3, 64, 64), "actions": torch.rand(4, 4, 2), "future_image": torch.rand(4, 3, 64, 64)}
    loader = [batch]
    model = JEPAModel()
    loss = train_epoch(model, loader, torch.optim.Adam(model.parameters(), lr=1e-3), 0.99)
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, {"seed": 1}, {"loss": loss})
    loaded = load_checkpoint(path, JEPAModel())
    assert math.isfinite(loss)
    assert loaded["metrics"]["loss"] == loss
```

- [ ] **步骤 2：运行测试，确认失败**

Run: `pytest -q tests/test_train.py`

Expected: FAIL，显示 `push_t_jepa.train` 不存在。

- [ ] **步骤 3：实现训练循环和持久化**

训练函数清零梯度、调用模型、计算 `torch.nn.functional.mse_loss`、反向传播、优化器步进并更新 EMA；验证使用 `torch.no_grad()`。检查点键固定为 `model_state`、`config`、`metrics`；缺失文件抛 `FileNotFoundError("检查点不存在: ...")`。

- [ ] **步骤 4：运行测试，确认通过**

Run: `pytest -q tests/test_train.py`

Expected: PASS，1 项通过。

- [ ] **步骤 5：提交**

```bash
git add push_t_jepa/train.py tests/test_train.py
git commit -m "feat: add JEPA training and checkpoints"
```

## 任务 6：CEM 与滚动时域规划

**文件：**
- Create: `push_t_jepa/planner.py`
- Test: `tests/test_planner.py`

**接口：**
- 消费：`JEPAModel`、`CEMConfig`、当前和目标 `np.ndarray[64,64,3]`。
- 产生：`CEMPlanner(model, config, seed)`；`plan(current_image, target_image) -> np.ndarray[horizon,2]`；`rollout_replan(env, target_image, steps) -> list[np.ndarray]`。
- `plan` 返回值被裁剪至 `[-1, 1]`；代价为模型预测与目标目标编码 embedding 的平方 L2 距离加 `1e-3 * sum(actions**2)`。

- [ ] **步骤 1：编写失败测试**

```python
import numpy as np
from push_t_jepa.model import JEPAModel
from push_t_jepa.planner import CEMPlanner

def test_cem_plan_has_bounded_action_sequence():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    actions = CEMPlanner(JEPAModel(), seed=5).plan(image, image)
    assert actions.shape == (8, 2)
    assert np.all(actions >= -1.0) and np.all(actions <= 1.0)
```

- [ ] **步骤 2：运行测试，确认失败**

Run: `pytest -q tests/test_planner.py`

Expected: FAIL，显示 `push_t_jepa.planner` 不存在。

- [ ] **步骤 3：实现 CEM**

使用局部 `np.random.Generator` 从每维正态分布采样、裁剪动作、向量化为 torch 张量后调用模型的上下文/预测/目标编码接口。CEM horizon 必须是模型动作预测跨度的整数倍；每个 4 步块以 `predict_from_context` 接续上一个预测 embedding，从而得到 8 步终点预测。每轮以 `np.argpartition` 选精英，更新均值和标准差并保留最小 `0.05` 标准差。滚动函数每次规划、执行第一个动作、记录新观察。

- [ ] **步骤 4：写固定环境的随机基线比较测试并运行**

```python
from push_t_jepa.env import PushTEnv

def test_rollout_replanning_returns_one_observation_per_execution_step():
    env = PushTEnv(seed=9)
    observations = CEMPlanner(JEPAModel(), seed=9).rollout_replan(env, env.reset(), steps=3)
    assert len(observations) == 4
```

Run: `pytest -q tests/test_planner.py`

Expected: PASS，2 项通过。另增加端到端比较：以一个可注入的确定性预测器替代 `JEPAModel`，断言 CEM 的动作代价小于同种子 128 条随机序列中的中位数，避免要求未训练模型在真实环境获胜。

- [ ] **步骤 5：提交**

```bash
git add push_t_jepa/planner.py tests/test_planner.py
git commit -m "feat: add embedding-space CEM planner"
```

## 任务 7：命令行演示、smoke 测试和文档

**文件：**
- Create: `push_t_jepa/demo.py`
- Create: `tests/test_smoke.py`
- Create: `README.md`
- Modify: `push_t_jepa/train.py`

**接口：**
- 消费：训练检查点、`PushTEnv` 和 `CEMPlanner`。
- 产生：`python -m push_t_jepa.train --smoke --output artifacts/smoke`；`python -m push_t_jepa.demo --checkpoint artifacts/smoke/model.pt --output artifacts/demo --seed 7`；演示目录中的 `rollout.gif` 与 `metrics.json`。
- smoke 训练使用 8 条、每条 8 步的轨迹和 1 个 epoch，必须在 CPU 完成。

- [ ] **步骤 1：编写失败端到端测试**

```python
from push_t_jepa.train import run_smoke_training
from push_t_jepa.demo import run_demo

def test_smoke_training_and_demo_write_reusable_artifacts(tmp_path):
    checkpoint = run_smoke_training(tmp_path / "train", seed=4)
    output = run_demo(checkpoint, tmp_path / "demo", seed=4, steps=3)
    assert checkpoint.is_file()
    assert (output / "rollout.gif").is_file()
    assert (output / "metrics.json").is_file()
```

- [ ] **步骤 2：运行测试，确认失败**

Run: `pytest -q tests/test_smoke.py`

Expected: FAIL，显示 `run_smoke_training` 或 `push_t_jepa.demo` 不存在。

- [ ] **步骤 3：实现 CLI 与产物导出**

为训练和演示添加 `argparse` 入口；用 Pillow 的 `Image.save(..., save_all=True, append_images=..., duration=...)` 写 GIF；将种子、初末几何距离、embedding 距离和执行步数写入 UTF-8 JSON。README 给出创建虚拟环境、安装、`pytest -q`、smoke、正式训练和演示的完整命令，并说明 JEPA 训练目标、EMA 与 CEM 的关系。

- [ ] **步骤 4：运行端到端测试与全量回归**

Run: `pytest -q tests/test_smoke.py && pytest -q`

Expected: PASS；`rollout.gif` 和 `metrics.json` 都存在，完整测试集无失败。

- [ ] **步骤 5：提交**

```bash
git add push_t_jepa/train.py push_t_jepa/demo.py tests/test_smoke.py README.md
git commit -m "feat: add runnable JEPA CEM demo"
```

## 最终验证

- [ ] 在干净虚拟环境中运行 `python -m pip install -e '.[dev]'`。
- [ ] 运行 `pytest -q` 并记录全部通过的输出。
- [ ] 运行 `python -m push_t_jepa.train --smoke --output artifacts/smoke --seed 7`。
- [ ] 运行 `python -m push_t_jepa.demo --checkpoint artifacts/smoke/model.pt --output artifacts/demo --seed 7`。
- [ ] 检查 `artifacts/demo/rollout.gif` 与 `artifacts/demo/metrics.json`，确认指标为有限数值、GIF 包含多帧，并在 README 中保留命令。

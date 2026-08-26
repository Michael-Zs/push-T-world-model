# Pymunk Push-T 迁移实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 以 Pymunk 刚体动力学替换手写 Push-T 接触规则，并保持 JEPA、CEM 和参数化图像尺寸工作流可运行。

**Architecture:** `PushTEnv` 维护 Pymunk 空间、运动学圆推杆、复合 T 动态刚体和静态边界；动作转为推杆目标速度并运行固定子步。Pillow 从 Pymunk 状态离屏渲染，现有数据集、训练和规划继续只调用环境公共接口。

**Tech Stack:** Python 3.10+、Pymunk、NumPy、PyTorch、Pillow、pytest。

**Spec:** `docs/superpowers/specs/2026-08-26-pymunk-push-t-design.md`

## Global Constraints

- 环境必须不依赖 Pygame 窗口；渲染返回 `np.uint8` RGB 数组。
- 保留归一化二维动作、`reset`、`step`、`render`、`state`、`set_state` 公共接口。
- 物理时间步固定为 `1/240`，每个环境动作执行 8 个子步。
- `image_size` 必须支持 64 与 256，且训练 CLI 的值和模型配置一致。
- 迁移后所有训练数据均须从 Pymunk 环境重新采集。

---

### 任务 1：依赖与物理配置

**文件：**
- Modify: `pyproject.toml`
- Modify: `push_t_jepa/config.py`
- Test: `tests/test_config.py`

**接口：**
- 产生：`EnvConfig(physics_dt=1/240, physics_substeps=8, pusher_speed=1.0, object_friction=0.8)`。
- `pymunk>=6.8` 是运行时依赖；非法物理参数抛中文 `ValueError`。

- [ ] **步骤 1：写失败测试**

```python
def test_physics_configuration_has_fixed_substeps():
    config = EnvConfig()
    assert config.physics_substeps == 8
    assert config.physics_dt == 1 / 240
```

- [ ] **步骤 2：运行并确认测试失败**

Run: `pytest -q tests/test_config.py::test_physics_configuration_has_fixed_substeps`

Expected: FAIL，`EnvConfig` 没有物理参数。

- [ ] **步骤 3：实现配置与依赖**

在 `EnvConfig` 添加物理字段和 `__post_init__` 校验；在 `pyproject.toml` 添加 Pymunk。

- [ ] **步骤 4：运行配置测试**

Run: `pytest -q tests/test_config.py`

Expected: PASS。

### 任务 2：Pymunk 环境与物理回归

**文件：**
- Modify: `push_t_jepa/env.py`
- Modify: `tests/test_env.py`

**接口：**
- 消费：`EnvConfig`。
- 产生：现有 `PushTEnv` 接口，内部改为 `pymunk.Space`。
- `step(action)` 返回 `(image, PushTState)`；`pusher_clearance()` 返回非负碰撞间隙。

- [ ] **步骤 1：写失败测试**

```python
def test_off_center_pymunk_push_rotates_t_object():
    env = PushTEnv(EnvConfig(image_size=64), seed=0)
    env.reset()
    env.set_state(np.array([0.36, 0.55]), np.array([0.50, 0.50]), 0.0)
    for _ in range(5):
        env.step(np.array([1.0, 0.0]))
    assert abs(env.state.object_angle) > 0.05
```

- [ ] **步骤 2：运行并确认旧规则不满足**

Run: `pytest -q tests/test_env.py::test_off_center_pymunk_push_rotates_t_object`

Expected: FAIL，旧近似接触不能达到角度阈值或不使用 Pymunk。

- [ ] **步骤 3：实现 Pymunk 空间**

创建零重力 `Space`，加入静态墙、运动学圆推杆和由两矩形 shape 组成的单一动态 T body。`step` 对动作设置推杆速度，调用 8 次 `space.step(dt)`，然后清零推杆速度。

- [ ] **步骤 4：实现 Pillow 渲染与测试辅助状态设置**

从 Pymunk body 绘制旋转 T 和圆推杆；`set_state` 更新位置、角度和速度后执行 `space.reindex_shapes_for_body`。

- [ ] **步骤 5：运行环境回归**

Run: `pytest -q tests/test_env.py`

Expected: PASS，包含确定性、边界、无穿透、正向接触与偏心旋转。

### 任务 3：参数化尺寸工作流与端到端验证

**文件：**
- Modify: `push_t_jepa/dataset.py`
- Modify: `push_t_jepa/train.py`
- Modify: `push_t_jepa/demo.py`
- Test: `tests/test_dataset.py`
- Test: `tests/test_smoke.py`

**接口：**
- 消费：`--image-size`。
- 产生：训练检查点的 `config.image_size`，演示加载同尺寸模型，GIF 宽度为 `3 * image_size`。

- [ ] **步骤 1：增加 256 像素端到端失败测试**

```python
def test_256_smoke_training_and_demo_use_matching_images(tmp_path):
    checkpoint = run_training(tmp_path / "train", trajectories=2, steps=8, epochs=1, batch_size=2, image_size=256)
    output = run_demo(checkpoint, tmp_path / "demo", steps=1)
    assert Image.open(output / "rollout.gif").size == (768, 256)
```

- [ ] **步骤 2：运行并确认失败**

Run: `pytest -q tests/test_smoke.py::test_256_smoke_training_and_demo_use_matching_images`

Expected: FAIL，直到 Pymunk 环境和尺寸配置均贯通。

- [ ] **步骤 3：修正调用与说明**

确保采集显式使用训练尺寸，演示根据 checkpoint 构造环境与模型；README 添加 Pymunk 安装与重新训练命令。

- [ ] **步骤 4：完整验证**

Run: `pytest -q && python -m push_t_jepa.train --smoke --output artifacts/pymunk-smoke --seed 7 && python -m push_t_jepa.demo --checkpoint artifacts/pymunk-smoke/model.pt --output artifacts/pymunk-demo --seed 7`

Expected: PASS；Pymunk 新数据训练完成，演示 GIF 与指标文件存在。

### 任务 4：提交与迁移说明

**文件：**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-08-26-pymunk-push-t-design.md`

- [ ] **步骤 1：记录迁移影响**

README 明确旧 checkpoint 不可作为 Pymunk 正式规划模型使用，必须重新采集和训练。

- [ ] **步骤 2：最终检查**

Run: `git diff --check && pytest -q`

Expected: 无空白错误、全绿。

- [ ] **步骤 3：提交**

```bash
git add pyproject.toml push_t_jepa tests README.md docs/superpowers
git commit -m "feat: replace Push-T dynamics with Pymunk"
```

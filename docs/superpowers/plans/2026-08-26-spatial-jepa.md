# Spatial JEPA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Push-T JEPA 从全局向量表征改为可保留物体位置的 8x8 空间表征。

**Architecture:** 图像编码器产生固定大小的空间 feature map；卷积预测器接收广播后的动作条件并预测未来 feature map；decoder 和 CEM 在相同空间 latent 上工作。

**Tech Stack:** Python、PyTorch、NumPy、pytest。

**Spec:** `docs/superpowers/specs/2026-08-26-spatial-jepa-design.md`

## Global Constraints

- 保持 CPU 优先，默认图像大小为 64，仍支持 256。
- `embedding_dim` 表示空间 latent 通道数，空间尺寸为固定 8。
- 旧向量检查点必须报出需要重新训练的明确错误。

---

### Task 1: 空间模型接口

**Files:**
- Modify: `tests/test_model.py`
- Modify: `push_t_jepa/config.py`
- Modify: `push_t_jepa/model.py`

**Interfaces:**
- Produces: `encode_context(image) -> Tensor[B,C,8,8]` 与 `decode(latent) -> Tensor[B,3,S,S]`。

- [ ] **Step 1: 写失败测试**：断言默认模型的 prediction/target 是 `(B,64,8,8)`，decoder 从同形状 latent 生成 64/256 RGB 图；断言错误 latent 被拒绝。
- [ ] **Step 2: 运行失败测试**：`pytest tests/test_model.py -q`，预期旧向量接口不满足新形状。
- [ ] **Step 3: 实现最小空间编码器、卷积动作预测器和空间 decoder**。
- [ ] **Step 4: 运行模型测试**：`pytest tests/test_model.py -q`。
- [ ] **Step 5: 提交**：`git add push_t_jepa/config.py push_t_jepa/model.py tests/test_model.py && git commit -m "feat: use spatial JEPA latents"`。

### Task 2: 训练与规划适配

**Files:**
- Modify: `tests/test_planner.py`
- Modify: `push_t_jepa/train.py`
- Modify: `push_t_jepa/planner.py`

**Interfaces:**
- Consumes: `Tensor[B,C,8,8]` 空间 latent。
- Produces: 空间维度目标距离的 CEM 动作序列。

- [ ] **Step 1: 写失败的空间确定性 CEM 测试**。
- [ ] **Step 2: 运行失败测试**：`pytest tests/test_planner.py -q`。
- [ ] **Step 3: 在训练方差统计与 CEM 代价中覆盖空间维度，并更新测试替身模型。**
- [ ] **Step 4: 运行训练和规划测试**：`pytest tests/test_train.py tests/test_planner.py -q`。
- [ ] **Step 5: 提交**：`git add push_t_jepa/train.py push_t_jepa/planner.py tests/test_planner.py && git commit -m "feat: plan in spatial JEPA latent space"`。

### Task 3: 端到端验证与远程复现

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 写出重新训练要求与测试命令。**
- [ ] **Step 2: 运行完整测试与 smoke 训练**：`pytest -q`，`python -m push_t_jepa.train --smoke --output /tmp/push-t-spatial-smoke`，然后运行 demo。
- [ ] **Step 3: 提交、推送，并在 `zhangzonggang@mac-mini` 拉取后运行 smoke 训练与 demo。**

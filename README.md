# CPU 优先的 Push-T JEPA + CEM

这是一个自包含的最小复现实验：手写二维 Push-T 仿真生成 64×64 图像，JEPA 从当前图像和未来 4 步动作预测未来 embedding；CEM 搜索 8 步动作序列，使预测终点 embedding 接近目标图像 embedding。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

## 自测与 smoke 演示

```bash
pytest -q
python -m push_t_jepa.train --smoke --output artifacts/smoke --seed 7
python -m push_t_jepa.demo --checkpoint artifacts/smoke/model.pt --output artifacts/demo --seed 7
```

演示会生成 `artifacts/demo/rollout.gif` 与 `artifacts/demo/metrics.json`。GIF 展示真实环境中滚动规划所执行的动作；JSON 记录目标的初始与最终几何距离。

## 结构

- `push_t_jepa/env.py`：无 GUI 的 Pillow 渲染、近似接触推动与旋转。
- `push_t_jepa/dataset.py`：带种子的随机轨迹及 JEPA 样本。
- `push_t_jepa/model.py`：上下文 CNN、EMA 目标 CNN 与动作条件预测器。
- `push_t_jepa/train.py`：embedding MSE 训练、EMA 更新和检查点。
- `push_t_jepa/planner.py`：在 embedding 距离上优化的 CEM；8 步候选由两段 4 步 latent 预测递推得到终点。

## 解释

JEPA 不重建目标像素，而是最小化 `预测未来 embedding` 与 `EMA 目标编码器产生的未来 embedding` 的均方误差。规划时，目标图像先被编码为目标 embedding，CEM 反复采样动作序列、用 JEPA 预测终点 embedding，并保留距离更小的精英动作序列来更新采样分布。

该仓库的物理是为理解表征学习与模型预测控制而刻意简化的近似，不是严格的真实摩擦仿真。Smoke 模式仅验证从数据、训练到规划产物的完整链路；要评估成功率，应增加训练轨迹和 epoch，并与随机动作基线比较最终几何距离。

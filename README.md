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

演示会生成 `artifacts/demo/rollout.gif` 与 `artifacts/demo/metrics.json`。GIF 每帧从左到右依次为：真实环境执行画面、JEPA 预测 embedding 经 decoder 解出的图像、目标图像；JSON 记录目标的初始与最终几何距离及 embedding 距离。

## 结构

- `push_t_jepa/env.py`：无 GUI 的 Pillow 渲染、近似接触推动与旋转。
- `push_t_jepa/dataset.py`：带种子的随机轨迹及 JEPA 样本。
- `push_t_jepa/model.py`：上下文 CNN、EMA 目标 CNN 与动作条件预测器。
- `push_t_jepa/train.py`：embedding MSE 训练、EMA 更新和检查点。
- `push_t_jepa/planner.py`：在 embedding 距离上优化的 CEM；8 步候选由两段 4 步 latent 预测递推得到终点。

## 解释

JEPA 的主目标是最小化 `预测未来 embedding` 与 `EMA 目标编码器产生的未来 embedding` 的均方误差。额外的轻量 decoder 从 embedding 重建图像，只用于让训练信号和演示可视化更直观；CEM 仍然只使用 embedding 距离。规划时，目标图像先被编码为目标 embedding，CEM 反复采样动作序列、用 JEPA 预测终点 embedding，并保留距离更小的精英动作序列来更新采样分布。

该仓库的物理是为理解表征学习与模型预测控制而刻意简化的近似，不是严格的真实摩擦仿真。Smoke 模式仅验证从数据、训练到规划产物的完整链路；要评估成功率，应增加训练轨迹和 epoch，并与随机动作基线比较最终几何距离。

## 正式 CPU 训练

先以中等规模确认机器内存与耗时：

```bash
python -m push_t_jepa.train \
  --output artifacts/train-medium --seed 7 \
  --trajectories 1000 --steps 24 --epochs 30 --batch-size 64 \
  --learning-rate 3e-4 --variance-weight 0.1
python -m push_t_jepa.demo \
  --checkpoint artifacts/train-medium/model.pt \
  --output artifacts/demo-medium --seed 7 --steps 20
```

确认 `history.json` 中验证 MSE 稳定下降后，再运行默认规模：

```bash
python -m push_t_jepa.train --output artifacts/train --seed 7
python -m push_t_jepa.demo --checkpoint artifacts/train/model.pt --output artifacts/demo --seed 7 --steps 24
```

训练数据中 70% 的动作朝 T 物体移动以提高接触样本比例，30% 保持随机探索。训练损失是归一化 embedding 的预测 MSE 加方差下界正则；后者用于抑制所有图像被编码为同一个向量的坍塌。

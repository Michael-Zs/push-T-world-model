# CPU 优先的 Push-T JEPA + CEM

这是一个自包含的最小复现实验：手写二维 Push-T 仿真生成 64×64 图像，JEPA 从当前图像和未来 4 步动作预测未来的 `8×8` 空间 latent；CEM 搜索 8 步动作序列，使预测终点 latent 接近目标图像 latent。

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
- `push_t_jepa/model.py`：输出 `通道×8×8` 空间 latent 的上下文 CNN、EMA 目标 CNN 与动作条件卷积预测器。
- `push_t_jepa/train.py`：embedding MSE 训练、EMA 更新和检查点。
- `push_t_jepa/planner.py`：先由渲染图像中的推杆/T 颜色引导推杆接触，再在空间 latent 距离上使用 CEM；8 步候选由两段 4 步 latent 预测递推得到终点。

## 解释

JEPA 的主目标是最小化 `预测未来空间 latent` 与 `EMA 目标编码器产生的未来空间 latent` 的均方误差。相比全局平均池化的单向量，`8×8` 布局会保留推杆和 T 物体大致在哪里；额外的轻量 decoder 从该空间 latent 重建图像，用于让训练信号和演示可视化更直观。CEM 同样只使用空间 latent 距离。

该版本与旧的全局向量模型结构不兼容；更新代码后必须重新训练，不能复用以前的 `model.pt`。

该仓库的物理是为理解表征学习与模型预测控制而刻意简化的近似，不是严格的真实摩擦仿真。环境的每步位移被限制为 `0.01`；因此 demo 在推杆未接触 T 前使用图像几何引导，接触后才将控制交给 CEM。Smoke 模式仅验证从数据、训练到规划产物的完整链路；要评估成功率，应增加训练轨迹和 epoch，并与随机动作基线比较最终几何距离。

## 正式 CPU 训练

先以中等规模确认机器内存与耗时：

```bash
python -m push_t_jepa.train \
  --output artifacts/train-medium --seed 7 \
  --trajectories 1000 --steps 64 --epochs 30 --batch-size 256 \
  --threads 10 --action-horizon 8 \
  --learning-rate 3e-4 --variance-weight 0.1
python -m push_t_jepa.demo \
  --checkpoint artifacts/train-medium/model.pt \
  --output artifacts/demo-medium --seed 7 --steps 20
```

`--threads` 应设为机器的逻辑核心数（这台 Mac 为 10）；`--batch-size 256` 是 16 GB Mac 的起始值。它会显著减少每个 epoch 的 CPU 调度开销，但不要求也不应刻意占满内存。若训练损失波动明显，先降回 `128`；若机器内存不足，再降为 `64`。

确认 `history.json` 中验证 MSE 稳定下降后，再运行默认规模：

```bash
python -m push_t_jepa.train --output artifacts/train --seed 7
python -m push_t_jepa.demo --checkpoint artifacts/train/model.pt --output artifacts/demo --seed 7 --steps 24
```

训练数据中 70% 的动作朝 T 物体移动以提高接触样本比例，30% 保持随机探索。训练损失是归一化 embedding 的预测 MSE 加方差下界正则；后者用于抑制所有图像被编码为同一个向量的坍塌。

## VAE-JEPA 对照实验

`--vae` 会把上下文编码器替换为空间 VAE：它为每个 `8×8` latent 位置输出均值和方差，训练时从该分布采样，并联合优化 JEPA、位姿、前景重建与 KL 正则。EMA 目标编码器始终使用均值，因而规划与评估是确定性的。KL 权重会在前几轮从零线性升到设定值。

```bash
python -m push_t_jepa.train \
  --vae --output artifacts/train-vae-h8 --seed 7 \
  --trajectories 1000 --steps 64 --epochs 20 --batch-size 128 \
  --threads 10 --action-horizon 8 \
  --kl-weight 0.001 --kl-warmup-epochs 5
python -m push_t_jepa.evaluate \
  --checkpoint artifacts/train-vae-h8/model.pt \
  --output artifacts/eval-vae-h8 --steps 80
```

TensorBoard 会额外给出 `vae/kl_loss`、`vae/reconstruction_loss`、`validation/pusher_pose_mse` 与 `validation/t_pose_mse`。`demo` 和 `evaluate` 会从 checkpoint 的 `model_type` 自动选择 VAE 或普通 JEPA，旧 checkpoint 没有该字段时仍按普通 JEPA 处理；两种模型参数不可以用 `--resume` 互相恢复。

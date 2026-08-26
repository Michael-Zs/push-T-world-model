# VAE 空间编码器 JEPA 实验设计

## 目标

在独立分支中验证 VAE 空间编码器能否让 Push-T latent 更清楚地保留推杆、T 物体及其姿态，同时不降低 JEPA+CEM 的真实控制效果。

## 架构

输入图像经过 VAE 编码器，产生空间均值与对数方差；重参数化后的 `64×8×8` latent 供 JEPA predictor 和 pose head 使用。VAE decoder 从该 latent 重建当前图像。EMA target encoder 对未来图像编码，并提供 JEPA 的预测目标。

```text
image ──> VAE encoder ──> z ──> action predictor ──> future z
                  │                 │
                  ├─> VAE decoder   └─> pose head
                  │
                  └─> KL regularization
```

## 训练目标

总损失由四部分组成：

1. JEPA future-latent 预测损失；
2. T 与推杆的位姿监督损失；
3. VAE 前景加权重建损失；
4. KL 正则，权重从零线性 warm-up，避免早期 posterior collapse。

decoder 的重建梯度会进入 VAE encoder；这是本实验相对当前 decoder-only 读出头的关键改变。通过独立分支与固定 seed、多 seed CEM 评估判断其是否值得保留。

## 兼容性与接口

- 新模型使用新的 `VAEModelConfig` 标记，旧 checkpoint 不兼容；
- demo、评估和 CEM 保持原接口，按 checkpoint config 自动构造 VAE-JEPA 模型；
- `--oracle-cost` 继续作为物理上限基线，不参与训练。

## 验证

1. 单元测试：latent 形状、KL 非负、重参数化与 checkpoint round trip；
2. smoke training：采集、训练、demo 全链路可运行；
3. 训练诊断：TensorBoard 记录重建、KL、JEPA、推杆/T 位姿损失；
4. 控制评估：固定 5-seed、80-step CEM 对比当前 large 模型，使用平均最终几何距离作为主指标；
5. 可视化：比较 `decode(encode(image))`，确认推杆与 T 均可辨。

## 成功标准

- 5-seed 平均最终距离不劣于当前 large 模型；
- 自编码图能清楚定位推杆与 T；
- KL 保持有限且非零，训练不坍塌。

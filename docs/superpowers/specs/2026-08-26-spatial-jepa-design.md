# Push-T 空间 JEPA 设计

## 目标

让视觉表征保留推杆与 T 物体的空间位置和朝向，使动作条件预测与 CEM 规划能够依据相对几何关系工作。

## 问题

现有编码器将卷积特征经 `AdaptiveAvgPool2d(1)` 汇聚为单个向量。全局平均会抹去物体在哪里；因此低验证 MSE 不能证明模型学会了可控的视觉动力学，decoder 也只能产生模糊平均图像。

## 决策

使用固定 `8x8` 的空间 latent：编码器输出 `[B, latent_channels, 8, 8]`，并逐空间位置做 L2 归一化。预测器将整段动作编码成通道条件，广播后交给卷积残差块，输出同形状未来 latent。EMA target encoder 保持不变。

`embedding_dim` 继续表示 latent 通道数，以保持训练参数含义简洁；新增 `spatial_size=8` 配置。decoder 直接从空间 latent 上采样到配置图像尺寸，不再使用全连接投影。

## 数据流

`image -> spatial context -> action-conditioned convolutional predictor -> predicted spatial latent`

训练的 JEPA MSE、方差正则与 CEM 目标距离都对通道、高、宽三个维度计算。方差统计将 `[B,C,H,W]` 展平为位置样本 `[B*H*W,C]`。重建头仍只服务于表征诊断和辅助训练。

## 兼容性与错误处理

旧的向量 latent 检查点和新结构参数形状不兼容，加载时应明确拒绝，并提示必须重新训练。图像尺寸仍限定为不小于 64 的 2 的幂；空间尺寸固定为 8。调用 `decode` 和 `predict_from_context` 时必须验证空间 latent 形状。

## 验证

单元测试应覆盖：64 和 256 输入输出形状、空间 latent 不被池化、target 无梯度、错误 latent 拒绝，以及用空间 latent 的确定性模型验证 CEM 朝目标方向优化。全套 pytest、smoke 训练和生成 GIF 是提交前门槛。

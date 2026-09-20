# 测试

运行 `python -m unittest discover -s tests -v`。当前 18 项通过，覆盖权重损坏检测、采样与梯度、BCE/Dice 反向、padding、短语聚合、原版 AR 数值、权重严格加载与预测一致性、PNG 标注转换和图片/文本完整前向。

模型测试用随机参数和合成样本，测试不会执行 optimizer.step。真实论文指标仍需完好权重和数据验证。

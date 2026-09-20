# Deformable attention

当前采用官方 Deformable-DETR 的纯 PyTorch 参考核，已通过采样数值和梯度检查；无需编译 CUDA。属于重建依赖，原 CUDA 算子缺失，尚未验证两者的数值一致性。来源及许可见 docs/算子恢复说明.md。

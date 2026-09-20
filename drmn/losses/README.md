# 损失

统一导出 CrossEntropyLoss(use_sigmoid=True) 和 DiceLoss。原实现位于 models/head_model；runtime.stage_loss 在每阶段对有效 noun token 累加 BCE + Dice，不让 padding 参与损失。具体最终训练入口已损坏，因此当前训练链属于依据保存代码和论文重建，未声称逐字恢复该入口。

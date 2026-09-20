"""原始 BCE/Dice 损失的统一导出，不改变计算。"""
from drmn.models.head_model.cross_entropy_loss import CrossEntropyLoss
from drmn.models.head_model.dice_loss import DiceLoss

__all__ = ["CrossEntropyLoss", "DiceLoss"]

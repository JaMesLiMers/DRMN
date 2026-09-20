# Losses

This package exports `CrossEntropyLoss(use_sigmoid=True)` and `DiceLoss` from `drmn/models/head_model`. `drmn.runtime.stage_loss` sums BCE + Dice over stages for valid noun tokens, excluding padding.

The archived Dice formula is retained. Its difference from the paper is documented in the [implementation notes](../../docs/consistency-audit.md#dice-loss).

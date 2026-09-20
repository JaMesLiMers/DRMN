# Losses

This package exports `CrossEntropyLoss(use_sigmoid=True)` and `DiceLoss` from the archived implementations under `drmn/models/head_model`. `drmn.runtime.stage_loss` sums BCE + Dice over stages for valid noun tokens, excluding padding.

The complete original training entry point was damaged. Current training workflows are reconstructed from saved code and the paper, not claimed as verbatim recovery of that entry point. The archived Dice formula is preserved; its difference from the printed paper formula is documented in the [consistency audit](../../docs/consistency-audit.md).

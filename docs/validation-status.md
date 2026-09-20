# Validation status and remaining work

Release constraints: do not rerun full training or prepare real datasets; a release without checkpoints is acceptable. Synthetic forward, loss, and backward checks are permitted without model parameter updates. See the [consistency audit](consistency-audit.md) for the current findings.

## Completed checks

- CPU environment installation and main-model imports.
- Random image plus tokenized text through frozen encoders to four-stage DRMN masks.
- Feature input through BCE + Dice to finite gradients without parameter updates.
- Archived state structure compatibility: 281 head, 538 image encoder, and 465 BERT entries.
- Identical four-stage core outputs before/after organization using the same reference operator.
- Prediction consistency after checkpoint save/load; rejection of missing parameters.
- Evaluation and mask-export CLIs using temporary random weights and synthetic data.
- All 26 unit and integration tests, including pretrained mapping, structural variants, analysis exports, and mocked save/resume control.
- Single-process and two-process CPU forward/backward validation, including distributed accumulation without updates.

## Outstanding performance validation

- Intact original weights: the available backup has 424 storage blocks with CRC failures, including 46 in the main head.
- Complete real PNG/COCO 2017 data, split, and preprocessing verification.
- Comparison between the missing original CUDA operator and the reconstructed reference implementation.
- Best-run settings, such as sampling-point count, that cannot be inferred from parameter shapes.
- Real predictions from original weights, the reported overall 62.9%, and grouped metrics.
- Actual public pretrained weight files, full epoch training, and GPU execution.

The archived backup has been turned into an executable engineering-validation release. It has not passed reproduction acceptance using original weights and real data. Missing weights do not automatically trigger retraining.

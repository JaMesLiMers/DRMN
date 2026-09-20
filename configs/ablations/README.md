# Ablation configurations

These are runnable variants for structural analysis, not verified reconstructions of the original experiment configurations or results.

- `no_image_encoder.yaml`: original archived projection-only encoder variant.
- `three_encoder_layers.yaml`: three image deformable encoder layers.
- `initial_matching_only.yaml`: initial response map without iterative refinement.

Each variant requires its own compatible head weights. Do not load the main-model checkpoint into a different architecture. Historical ablation results have not been reproduced with these configurations.

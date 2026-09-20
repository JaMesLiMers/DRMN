# Data preparation and format

See the [download guide](dataset-download.md) for official asset sources. This release does not include real data or a checkpoint.

## Raw inputs and preprocessing

Required inputs are COCO 2017 images, panoptic JSON, RGB-encoded panoptic PNG masks, and `png_coco_*_dataloader.json` containing `image_id`, `caption`, `noun_vector`, `labels`, and `boxes`. The recovered backup does not contain a complete dataset. Data from other projects is not substituted automatically.

Run `python tools/encode_data.py --help` for path arguments. The tool loads either an intact combined checkpoint or separate pretrained FPN/BERT weights before exporting features. A supplied combined checkpoint also undergoes strict DRMN head loading. Random weights are not accepted as substitutes for pretrained encoders. `noun_vector` must align with the original BERT WordPiece tokens; truncation that discards annotations raises an error. All valid records are processed by default. The archived visualization code's seven-batch filter is not used; `--limit` explicitly enables a smaller check.

Images are converted from RGB to BGR, resized using shortest-edge and maximum-size constraints, normalized, and padded to multiples of 32. Masks follow the original order: bilinear resize, padding, reduction to quarter resolution, and thresholding at `>0`. Evaluation uses this geometry. The original code's additional zero padding to 400×400 does not change nonempty-mask IoU.

## Frozen feature NPZ format

Each file represents one image and one narrative. Files are read with `allow_pickle=False`.

| Field | Shape and meaning |
|---|---|
| `fpn_p2`, `fpn_p3`, `fpn_p4`, `fpn_p5` | `[256,H,W]`; adjacent levels halve spatial dimensions; p2 dimensions are multiples of eight |
| `language` | `[max_sequence_length,768]` |
| `noun_ids` | `[max_sequence_length]`; zero denotes non-target tokens or padding, positive integers identify phrases; CLS/SEP positions are zero |
| `targets_token` | `[number_of_selected_noun_tokens,Hout,Wout]`; supervision targets |
| `phrase_intervals` | `[number_of_phrases+1]`; boundaries in the selected noun-token sequence, starting at zero |
| `targets_phrase` | `[number_of_phrases,Hout,Wout]`; evaluation targets |
| `ann_types` | `[number_of_phrases]`; 1 = singular, 2 = plural |
| `ann_categories` | `[number_of_phrases]`; 1 = thing, 2 = stuff |

`SOURCE.json` records provenance and whether the data is synthetic. Reusing frozen features requires consistent encoder weights, splits, and preprocessing.

The archived `test_features_compress.hdf5` fails object traversal with a `wrong B-tree signature` error and cannot be treated as a complete usable cache. Explicitly generated synthetic samples are used for engineering checks; their outputs are not paper evaluation results.

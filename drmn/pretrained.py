"""Load separate frozen encoder weights without a trained DRMN head."""
import codecs
import collections
import hashlib
import pickle
from pathlib import Path
import numpy as np
import torch
from drmn.models.frozen_encoders import FrozenImageEncoder, FrozenTextEncoder


class NumpyCheckpointReader(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ("_codecs", "encode"): codecs.encode,
            ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
            ("numpy.core.multiarray", "scalar"): np.core.multiarray.scalar,
            ("collections", "OrderedDict"): collections.OrderedDict,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Unsupported checkpoint object: {module}.{name}")
        return allowed[module, name]


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_weights(path):
    path = Path(path)
    if path.suffix == ".pkl":
        with path.open("rb") as stream:
            obj = NumpyCheckpointReader(stream, encoding="latin1").load()
    else:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    state = obj.get("state_dict", obj.get("model", obj))
    return {k.removeprefix("module."): torch.as_tensor(v) for k, v in state.items()
            if isinstance(v, (np.ndarray, torch.Tensor))}


def load_public_image(model, state):
    # Detectron2 model-zoo files omit these deterministic preprocessing buffers.
    state = dict(state)
    state.setdefault("pixel_mean", model.pixel_mean.clone())
    state.setdefault("pixel_std", model.pixel_std.clone())
    return model.load_original(state)


def load_public_text(model, state):
    state = {k.replace(".gamma", ".weight").replace(".beta", ".bias"): v
             for k, v in state.items()}
    complete = model.state_dict()
    loaded = []
    inactive = ("model.bert.encoder.visn_fc.", "model.bert.encoder.cross_layers.",
                "model.bert.encoder.single_layers.")
    for target, template in complete.items():
        if target.startswith(inactive):
            continue  # Archived text forward never calls these legacy branches.
        base = target.removeprefix("model.")
        candidates = (target, base, base.removeprefix("bert."))
        source = next((key for key in candidates if key in state), None)
        if source is None:
            raise ValueError(f"Missing active BERT parameter: {target}")
        value = state[source]
        if value.shape != template.shape or not torch.isfinite(value).all():
            raise ValueError(f"Invalid active BERT parameter: {target}")
        complete[target] = value
        loaded.append(target)
    model.load_state_dict(complete, strict=True)
    return {"active_loaded": len(loaded), "unused_legacy_entries": len(complete)-len(loaded)}


def load_encoders(args, cfg, checkpoint=None):
    image = FrozenImageEncoder()
    text = FrozenTextEncoder(args.bert_config, args.vocab, cfg["model"]["max_sequence_length"])
    if args.fpn_weights or args.bert_weights:
        if not (args.fpn_weights and args.bert_weights):
            raise ValueError("Supply both --fpn-weights and --bert-weights")
        load_public_image(image, read_weights(args.fpn_weights))
        load_public_text(text, read_weights(args.bert_weights))
        source = {"fpn_sha256": file_hash(args.fpn_weights), "bert_sha256": file_hash(args.bert_weights)}
    elif checkpoint is not None and "fpn_model_state" in checkpoint and "bert_model_state" in checkpoint:
        image.load_original(checkpoint["fpn_model_state"])
        text.load_original(checkpoint["bert_model_state"])
        source = {"combined_checkpoint_sha256": file_hash(args.checkpoint)}
    else:
        raise ValueError("Provide separate pretrained encoders or a combined original checkpoint")
    return image.to(args.device).eval(), text.to(args.device).eval(), source

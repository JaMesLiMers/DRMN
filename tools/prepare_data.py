"""Validate prepared feature samples or explicitly create a synthetic smoke fixture."""
if __package__ in (None, ""):
    import _bootstrap
import argparse,json
from pathlib import Path
import numpy as np
from drmn.runtime import read_config
from drmn.datasets.features import load_sample


def synthetic_fixture(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(0)
    data={f"fpn_p{i+2}":rng.standard_normal((256,s,s)).astype("float32") for i,s in enumerate([32,16,8,4])}
    target=(rng.random((2,32,32))>0.7).astype("float32")
    data.update(language=rng.standard_normal((8,768)).astype("float32"),noun_ids=np.array([0,1,2,0,0,0,0,0]),targets_token=target,targets_phrase=target,phrase_intervals=np.array([0,1,2]),ann_types=np.array([1,2]),ann_categories=np.array([1,2]))
    np.savez_compressed(directory/"synthetic_only.npz",**data)
    (directory/"SOURCE.json").write_text(json.dumps({"synthetic":True,"purpose":"pipeline validation, never paper metrics"},indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",required=True);p.add_argument("--data",required=True);p.add_argument("--synthetic-smoke",action="store_true")
    a=p.parse_args()
    if a.synthetic_smoke:synthetic_fixture(a.data)
    cfg=read_config(a.config);paths=sorted(Path(a.data).glob("*.npz"))
    if not paths:raise ValueError("No prepared samples")
    for path in paths:load_sample(path,cfg)
    print(f"Validated {len(paths)} samples. This does not certify official dataset coverage.")

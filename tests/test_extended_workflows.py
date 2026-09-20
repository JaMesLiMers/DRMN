import argparse
import importlib.util
import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from drmn.pretrained import read_weights,load_public_image,load_public_text
from drmn.models.frozen_encoders import FrozenTextEncoder
from drmn.training import epoch_batches,save_atomic
from drmn.runtime import build_model,read_config
from drmn.datasets.features import load_sample
from tools.prepare_data import synthetic_fixture
from tools.analyze import export_analysis

ROOT=Path(__file__).resolve().parents[1]

class ExtendedTests(unittest.TestCase):
    def test_public_numpy_pickle_loader_rejects_non_numpy_globals(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"weights.pkl"
            path.write_bytes(pickle.dumps({"model":{"weight":np.ones((2,3),dtype=np.float32)}},protocol=2))
            self.assertEqual(tuple(read_weights(path)["weight"].shape),(2,3))
            path.write_bytes(pickle.dumps(Path("unexpected")))
            with self.assertRaises(pickle.UnpicklingError):read_weights(path)

    def test_public_bert_covers_active_path_and_rejects_missing_weight(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as d:
            cfg=json.loads((ROOT/"configs/bert/bert_config.json").read_text())
            cfg.update(vocab_size=30522,hidden_size=24,num_attention_heads=4,intermediate_size=32)
            path=Path(d)/"bert.json";path.write_text(json.dumps(cfg))
            model=FrozenTextEncoder(path,ROOT/"configs/bert/vocab.txt",8)
            inactive=("model.bert.encoder.visn_fc.","model.bert.encoder.cross_layers.","model.bert.encoder.single_layers.")
            state={k.removeprefix("model."):v.clone() for k,v in model.state_dict().items() if not k.startswith(inactive)}
            result=load_public_text(model,state)
            self.assertEqual(result["active_loaded"],len(state))
            state.pop(next(iter(state)))
            with self.assertRaisesRegex(ValueError,"Missing active BERT"):load_public_text(model,state)

    def test_public_fpn_fills_only_preprocessing_buffers(self):
        class Example:
            pixel_mean=torch.zeros(3,1,1)
            pixel_std=torch.ones(3,1,1)
            def load_original(self,state):return state
        result=load_public_image(Example(),{"backbone.weight":torch.ones(1)})
        self.assertEqual(set(result),{"pixel_mean","pixel_std","backbone.weight"})

    def test_epoch_sharding_is_repeatable_and_padded_explicitly(self):
        first=epoch_batches(5,17,0,0,2,2)
        second=epoch_batches(5,17,0,1,2,2)
        self.assertEqual(first,epoch_batches(5,17,0,0,2,2))
        self.assertEqual([len(g) for g in first],[2,1])
        self.assertEqual(set(sum(first+second,[])),set(range(5)))
        self.assertEqual(len(sum(first+second,[])),6)
        self.assertNotEqual(first,epoch_batches(5,17,1,0,2,2))

    def test_epoch_save_best_resume_control_without_updates(self):
        # Exercise lifecycle with a tiny model and optimizer.step mocked: no training.
        sys.path.insert(0,str(ROOT/"tools"))
        spec=importlib.util.spec_from_file_location("epoch_cli",ROOT/"tools/train_epochs.py")
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        class Tiny(torch.nn.Module):
            def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(1.))
            def forward(self,*args):return [self.weight.square()],[]
        with tempfile.TemporaryDirectory() as d:
            data=Path(d)/"data";data.mkdir();(data/"x.npz").write_bytes(b"fixture")
            out=Path(d)/"run"
            sample={"fpn":[],"language":None,"noun_ids":None}
            cfg={"seed":0,"model":{},"training":{"learning_rate":.0001}}
            def restore(model,path):
                obj=torch.load(path,weights_only=True);model.load_state_dict(obj["model_state"]);return obj
            args=["train_epochs","--config","unused","--data",str(data),"--val-data",str(data),"--output",str(out),"--epochs","1"]
            with patch.object(module,"read_config",return_value=cfg),patch.object(module,"build_model",side_effect=lambda *a:Tiny()),patch.object(module,"load_sample",return_value=sample),patch.object(module,"stage_loss",side_effect=lambda pred,s:pred[0]),patch.object(module,"evaluate_files",return_value={"overall":{"average_recall":.4}}),patch.object(module,"load_checkpoint",side_effect=restore),patch.object(torch.optim.Adam,"step") as step:
                with patch.object(sys,"argv",args):module.main()
                self.assertEqual(step.call_count,1)
                checkpoint=torch.load(out/"last.pth",weights_only=True)
                self.assertEqual(checkpoint["completed_epochs"],1)
                self.assertEqual(checkpoint["model_state"]["weight"].item(),1.)
                self.assertTrue((out/"best.pth").exists())
                args[-1]="2";args.extend(["--resume",str(out/"last.pth")])
                with patch.object(sys,"argv",args):module.main()
                checkpoint=torch.load(out/"last.pth",weights_only=True)
                self.assertEqual(checkpoint["completed_epochs"],2)
                self.assertEqual(checkpoint["global_step"],2)
                self.assertEqual(checkpoint["model_state"]["weight"].item(),1.)
                self.assertEqual(len((out/"history.jsonl").read_text().splitlines()),2)

    def test_refinement_sampling_exports(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as d:
            data=Path(d)/"data";synthetic_fixture(data)
            cfg=read_config(ROOT/"configs/smoke.yaml")
            sample=load_sample(data/"synthetic_only.npz",cfg)
            model=build_model(cfg).eval()
            with torch.no_grad():predictions,debug=model(sample["fpn"],sample["language"],sample["noun_ids"])
            out=Path(d)/"figures";export_analysis(predictions,debug,sample,out)
            self.assertEqual(len(list(out.glob("*.png"))),7)
            with np.load(out/"diagnostics.npz",allow_pickle=False) as archive:
                self.assertIn("round_3_deformable_attention_weights",archive.files)
            self.assertEqual(json.loads((out/"analysis.json").read_text())["refinement_rounds"],3)

    def test_ablation_forward_backward(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as d:
            data=Path(d)/"data";synthetic_fixture(data)
            for path in sorted((ROOT/"configs/ablations").glob("*.yaml")):
                with self.subTest(config=path.name):
                    cfg=read_config(path)
                    cfg["model"].update(num_points=4,max_seg_num=2,max_sequence_length=8)
                    sample=load_sample(data/"synthetic_only.npz",cfg)
                    model=build_model(cfg)
                    outputs,_=model(sample["fpn"],sample["language"],sample["noun_ids"])
                    self.assertEqual(len(outputs),cfg["model"]["num_stages"]+1)
                    from drmn.runtime import stage_loss
                    loss=stage_loss(outputs,sample);loss.backward()
                    self.assertTrue(torch.isfinite(loss))

    def test_recall_plot_cli(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"result.json"
            path.write_text(json.dumps({"per_phrase":{"ious":[0.,.4,1.],"ann_types":[1,2,1],"ann_categories":[1,2,1]}}))
            out=Path(d)/"recall.png"
            subprocess.run([sys.executable,str(ROOT/"tools/plot_recall.py"),"--results",str(path),"--output",str(out)],check=True,capture_output=True)
            self.assertGreater(out.stat().st_size,1000)

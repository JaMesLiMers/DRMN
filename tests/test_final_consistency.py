import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch
import torch.nn.functional as F
from drmn.models.frozen_encoders import DetectronFrozenBatchNorm2d
from drmn.models.head_model.dice_loss import dice_loss

ROOT=Path(__file__).resolve().parents[1]

class FinalConsistencyTests(unittest.TestCase):
    def test_frozen_bn_matches_archived_inference_arithmetic(self):
        bn=DetectronFrozenBatchNorm2d(3)
        bn.running_mean.copy_(torch.tensor([.2,-.3,.1]))
        bn.running_var.copy_(torch.tensor([.7,1.2,.9]))
        bn.bias.copy_(torch.tensor([-.1,.2,.05]))
        x=torch.randn(2,3,7,9)
        expected=F.batch_norm(x,bn.running_mean,bn.running_var,bn.weight,bn.bias,training=False,eps=bn.eps)
        torch.testing.assert_close(bn(x),expected,rtol=0,atol=0)

    def test_preserved_dice_is_squared_not_paper_linear_formula(self):
        prediction=torch.tensor([[.2,.8]])
        target=torch.tensor([[0.,1.]])
        expected=1-(2*.8)/(.2**2+.8**2+1+2e-3)
        actual=dice_loss(prediction,target)
        self.assertAlmostEqual(actual.item(),expected,places=6)
        paper_linear=1-(2*.8)/(.2+.8+1)
        self.assertGreater(abs(actual.item()-paper_linear),.01)

    def test_official_annotation_to_dataloader_without_download(self):
        with tempfile.TemporaryDirectory() as directory:
            ann=Path(directory)/"annotations";ann.mkdir()
            panoptic={"images":[{"id":7}],"annotations":[{"image_id":7,"segments_info":[{"id":1,"bbox":[0,0,2,2]}]}]}
            raw=[{"image_id":"7","caption":"a black bear", "segments":[{"utterance":"a","segment_ids":[],"noun":False,"plural":False},{"utterance":"black bear","segment_ids":["1"],"noun":True,"plural":False}]}]
            (ann/"panoptic_val2017.json").write_text(json.dumps(panoptic))
            (ann/"png_coco_val2017.json").write_text(json.dumps(raw))
            result=subprocess.run([sys.executable,str(ROOT/"tools/preprocess_annotations.py"),"--data_dir",directory,"--splits","val2017","--vocab",str(ROOT/"configs/bert/vocab.txt")],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            record=json.loads((ann/"png_coco_val2017_dataloader.json").read_text())[0]
            self.assertEqual(record["noun_vector"],[0,1,1])
            self.assertEqual(record["labels"],[[-2],[-1],[-1]])
            self.assertEqual(record["boxes"],[[[0,0,0,0]],[[0,0,2,2]],[[0,0,2,2]]])

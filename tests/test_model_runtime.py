import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from drmn.runtime import build_model,read_config,forward_sample,stage_loss,load_checkpoint
from drmn.datasets.features import load_sample
from drmn.evaluation.metrics import average_recall,phrase_masks
from drmn.evaluation.meters import average_accuracy
from drmn.ops.reference import ms_deform_attn_core_pytorch
from tools.prepare_data import synthetic_fixture

ROOT=Path(__file__).resolve().parents[1]


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.cfg=read_config(ROOT/"configs/smoke.yaml")
        cls.temp=tempfile.TemporaryDirectory()
        synthetic_fixture(cls.temp.name)
        cls.sample=load_sample(Path(cls.temp.name)/"synthetic_only.npz",cls.cfg)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_sampling_pixel_centers_and_gradient(self):
        value=torch.tensor([1.,2.,3.,4.],dtype=torch.double).reshape(1,4,1,1).requires_grad_()
        locations=torch.tensor([[[[[[.25,.25],[.75,.75]]]]]],dtype=torch.double,requires_grad=True)
        weights=torch.tensor([[[[[.5,.5]]]]],dtype=torch.double,requires_grad=True)
        shapes=torch.tensor([[2,2]])
        output=ms_deform_attn_core_pytorch(value,shapes,locations,weights)
        torch.testing.assert_close(output,torch.tensor([[[2.5]]],dtype=torch.double))
        locations=torch.tensor([[[[[[.35,.4],[.65,.55]]]]]],dtype=torch.double,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda v,l,w:ms_deform_attn_core_pytorch(v,shapes,l,w),(value,locations,weights),eps=1e-6,atol=1e-4))

    def test_average_recall_matches_original(self):
        ious=np.array([0.,.0001,.12345,.5,.9999,1.])
        self.assertAlmostEqual(average_recall(ious),average_accuracy(ious),places=12)
        self.assertIsNone(average_recall([]))

    def test_phrase_aggregation_averages_probabilities(self):
        logits=torch.tensor([[[[4.]], [[-2.]], [[1.]]]])
        actual=phrase_masks(logits,[0,2,3])
        self.assertEqual(actual.tolist(),[[[True]],[[True]]])
        logits[0,0]=-4.
        self.assertFalse(phrase_masks(logits,[0,2,3])[0].item())

    def test_model_matches_all_archived_parameter_shapes(self):
        model=build_model(self.cfg)
        refs=json.loads((ROOT/"tests/fixtures/checkpoint_shapes.json").read_text())["tensor_references"]
        expected={r["path"].removeprefix("/model_state/module."):tuple(r["shape"]) for r in refs if r["path"].startswith("/model_state/module.")}
        self.assertEqual(expected,{k:tuple(v.shape) for k,v in model.state_dict().items()})

    def test_forward_loss_backward_no_optimizer_step(self):
        model=build_model(self.cfg)
        before=model.rpn_head.init_kernels.weight.detach().clone()
        predictions=forward_sample(model,self.sample)
        self.assertEqual(len(predictions),4)
        loss=stage_loss(predictions,self.sample)
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        grads=[p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertGreater(model.rpn_head.init_kernels.weight.grad.abs().sum().item(),0)
        torch.testing.assert_close(before,model.rpn_head.init_kernels.weight,rtol=0,atol=0)

    def test_eval_save_load_is_identical(self):
        model=build_model(self.cfg).eval()
        with torch.no_grad():before=forward_sample(model,self.sample)[-1]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"test_random_weights.pth"
            torch.save({"model_state":model.state_dict()},path)
            restored=build_model(self.cfg).eval()
            load_checkpoint(restored,path)
            with torch.no_grad():after=forward_sample(restored,self.sample)[-1]
            torch.testing.assert_close(before,after,rtol=0,atol=0)

    def test_missing_parameter_is_rejected(self):
        model=build_model(self.cfg)
        state=model.state_dict();state.pop(next(iter(state)))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"incomplete.pth";torch.save({"model_state":state},path)
            with self.assertRaises(RuntimeError):load_checkpoint(model,path)

    def test_padding_targets_do_not_affect_loss(self):
        prediction=torch.zeros(1,3,32,32,requires_grad=True)
        sample=dict(self.sample)
        sample["targets_token"]=torch.cat([sample["targets_token"],torch.ones(1,1,32,32)],1)
        first=stage_loss([prediction],sample)
        sample["targets_token"][:,-1]=0
        second=stage_loss([prediction],sample)
        torch.testing.assert_close(first,second)

if __name__=="__main__":unittest.main()

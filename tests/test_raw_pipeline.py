import json
from pathlib import Path
import unittest
import torch
from drmn.models.frozen_encoders import FrozenImageEncoder,FrozenTextEncoder
from drmn.runtime import build_model,read_config,forward_sample

ROOT=Path(__file__).resolve().parents[1]

class RawPipelineTests(unittest.TestCase):
    def test_encoder_mapping_and_image_caption_to_mask(self):
        torch.set_num_threads(2)
        image=FrozenImageEncoder()
        text=FrozenTextEncoder(ROOT/"configs/bert/bert_config.json",ROOT/"configs/bert/vocab.txt",8)
        refs=json.loads((ROOT/"tests/fixtures/checkpoint_shapes.json").read_text())["tensor_references"]
        expected={x["path"].removeprefix("/fpn_model_state/module."):tuple(x["shape"]) for x in refs if x["path"].startswith("/fpn_model_state/module.")}
        mapped={image.source_key(k):tuple(v.shape) for k,v in image.state_dict().items()}
        self.assertEqual(set(mapped),{k for k in expected if k.startswith("backbone.") or k in {"pixel_mean","pixel_std"}})
        self.assertTrue(all(expected[k]==v for k,v in mapped.items()))
        bert={x["path"].removeprefix("/bert_model_state/module."):tuple(x["shape"]) for x in refs if x["path"].startswith("/bert_model_state/module.")}
        self.assertEqual(bert,{k:tuple(v.shape) for k,v in text.state_dict().items()})
        self.assertFalse(any(p.requires_grad for p in image.parameters()))
        self.assertFalse(any(p.requires_grad for p in text.parameters()))
        caption="a man with a dog"
        self.assertEqual(text.tokenizer.tokenize(caption),["a","man","with","a","dog"])
        with torch.no_grad():
            f=image(torch.rand(3,128,128)*255);l=text(caption)
            model=build_model(read_config(ROOT/"configs/smoke.yaml")).eval()
            outputs=forward_sample(model,{"fpn":f,"language":l,"noun_ids":torch.tensor([[0,0,1,0,0,2,0,0]])})
        self.assertEqual(len(outputs),4)
        self.assertEqual(tuple(outputs[-1].shape),(1,2,32,32))
        self.assertTrue(torch.isfinite(outputs[-1]).all())

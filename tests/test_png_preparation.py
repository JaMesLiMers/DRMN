import unittest
import numpy as np
import torch
from drmn.datasets.png import annotation_targets,prepare_masks

class PNGTests(unittest.TestCase):
    def test_multitoken_phrase_and_plural_masks(self):
        record={"noun_vector":[1,1,0,2],"labels":[[7],[7],[-2],[8,9]],"boxes":[[[0,0,1,1]],[[0,0,1,1]],[[0,0,0,0]],[[1,0,1,1],[0,1,2,1]]]}
        ann={"segments_info":[{"id":7,"bbox":[0,0,1,1],"category_id":1},{"id":8,"bbox":[1,0,1,1],"category_id":2},{"id":9,"bbox":[0,1,2,1],"category_id":2}]}
        categories={1:{"isthing":True},2:{"isthing":False}}
        result=annotation_targets(record,ann,categories,np.array([[7,8],[9,9]]),8,4)
        self.assertEqual(result["phrase_intervals"].tolist(),[0,2,3])
        self.assertEqual(result["ann_types"].tolist(),[1,2])
        self.assertEqual(result["ann_categories"].tolist(),[1,2])
        self.assertEqual(result["original_masks"].sum((1,2)).tolist(),[1.,3.])
        masks=prepare_masks(result["original_masks"],(32,32))
        self.assertEqual(tuple(masks.shape),(2,8,8));self.assertTrue(torch.isfinite(masks).all())

    def test_positive_truncation_rejected(self):
        record={"noun_vector":[1]*9}
        with self.assertRaisesRegex(ValueError,"truncates"):
            annotation_targets(record,{}, {},np.zeros((2,2)),8,10)

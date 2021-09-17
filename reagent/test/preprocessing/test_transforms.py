import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from reagent.preprocessing import transforms


class TestTransforms(unittest.TestCase):
    def setUp(self):
        # preparing various components for qr-dqn trainer initialization
        # currently not needed
        pass

    def test_Compose(self):
        t1, t2 = Mock(return_value=2), Mock(return_value=3)
        compose = transforms.Compose(t1, t2)
        data = 1
        out = compose(data)
        t1.assert_called_with(1)
        t2.assert_called_with(2)
        self.assertEqual(out, 3)

    def test_ValuePresence(self):
        vp = transforms.ValuePresence()
        d1 = {"a": 1, "a_presence": 0, "b": 2}
        d2 = {"a_presence": 0, "b": 2}
        o1 = vp(d1)
        o2 = vp(d2)
        self.assertEqual(o1, {"a": (1, 0), "b": 2})
        self.assertEqual(o2, {"a_presence": 0, "b": 2})

    def test_MaskByPresence(self):
        keys = ["a", "b"]
        mbp = transforms.MaskByPresence(keys)
        data = {
            "a": (torch.tensor(1), torch.tensor(0)),
            "b": (torch.tensor(3), torch.tensor(1)),
        }
        expected = {"a": torch.tensor(0), "b": torch.tensor(3)}
        out = mbp(data)
        self.assertEqual(out["a"], expected["a"])
        self.assertEqual(out["b"], expected["b"])
        with self.assertRaisesRegex(Exception, "Not valid value"):
            data2 = {
                "a": torch.tensor(1),
                "b": (torch.tensor(3), torch.tensor(1)),
            }
            out = mbp(data2)
        with self.assertRaisesRegex(Exception, "Unmatching value shape"):
            data3 = {
                "a": (torch.tensor(1), torch.tensor([0, 2])),
                "b": (torch.tensor(3), torch.tensor(1)),
            }
            out = mbp(data3)

    def test_Lambda(self):
        lam = transforms.Lambda(keys=["a", "b", "c"], fn=lambda x: x + 1)
        data = {"a": 1, "b": 2, "c": 3, "d": 4}
        out = lam(data)
        self.assertEqual(out, {"a": 2, "b": 3, "c": 4, "d": 4})

    def test_SelectValuePresenceColumns(self):
        block = np.reshape(np.arange(16), (4, 4))
        data = {"a": (block, block + 16), "c": 1}
        svp = transforms.SelectValuePresenceColumns(
            source="a", dest="b", indices=[1, 2]
        )
        out = svp(data)
        expected = {
            "a": (block, block + 16),
            "b": (block[:, [1, 2]], block[:, [1, 2]] + 16),
            "c": 1,
        }
        for key in ["a", "b"]:
            self.assertTrue(np.all(out[key][0] == expected[key][0]))
            self.assertTrue(np.all(out[key][1] == expected[key][1]))
        self.assertEqual(out["c"], expected["c"])

    @patch("reagent.preprocessing.transforms.Preprocessor")
    def test_DenseNormalization(self, Preprocessor):
        a_out = torch.tensor(1)
        b_out = torch.tensor(2)
        c_out = torch.tensor(3.0)
        preprocessor = Mock(side_effect=[a_out, b_out])
        Preprocessor.return_value = preprocessor
        # of form (value, presence)
        a_in = (torch.tensor([1, torch.nan, 2]), torch.tensor([1, 1, 1]))
        b_in = (torch.tensor([1, 2, torch.nan]), torch.tensor([0, 1, 1]))
        data = {"a": a_in, "b": b_in, "c": c_out}
        normalization_data = Mock()
        dn = transforms.DenseNormalization(
            keys=["a", "b"], normalization_data=normalization_data
        )
        out = dn(data)
        self.assertEqual(out["a"], a_out.float())
        self.assertEqual(out["b"], b_out.float())
        # ensure unnamed variables not changed
        self.assertEqual(out["c"], c_out)
        in_1, in_2 = [call_args.args for call_args in preprocessor.call_args_list]
        self.assertTrue(torch.all(torch.stack(in_1) == torch.stack(a_in)))
        self.assertTrue(torch.all(torch.stack(in_2) == torch.stack(b_in)))

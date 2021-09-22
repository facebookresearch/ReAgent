import unittest
from typing import List
from unittest.mock import Mock, patch

import numpy as np
import reagent.core.types as rlt
import torch
from reagent.preprocessing import transforms
from reagent.preprocessing.types import InputColumn


class TestTransforms(unittest.TestCase):
    def setUp(self):
        # preparing various components for qr-dqn trainer initialization
        # currently not needed
        pass

    def assertDictComparatorEqual(self, a, b, cmp):
        """
        assertDictEqual() compares args with ==. This allows caller to override
        comparator via cmp argument.
        """
        self.assertIsInstance(a, dict, "First argument is not a dictionary")
        self.assertIsInstance(b, dict, "Second argument is not a dictionary")
        self.assertSequenceEqual(a.keys(), b.keys())

        for key in a.keys():
            self.assertTrue(cmp(a[key], b[key]), msg=f"Different at key {key}")

    def assertDictOfTensorEqual(self, a, b):
        """
        Helper method to compare dicts with values of type Tensor.

        Cannot use assertDictEqual when values are of type Tensor since
        tensor1 == tensor2 results in a tensor of bools. Use this instead.
        """

        def _tensor_cmp(a, b):
            return torch.all(a == b)

        self.assertDictComparatorEqual(a, b, _tensor_cmp)

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

    def test_StackDenseFixedSizeArray(self):
        # happy path: value is type Tensor; check cast to float
        value = torch.eye(4).to(dtype=torch.int)  # start as int
        data = {"a": value}
        out = transforms.StackDenseFixedSizeArray(data.keys(), size=4)(data)
        expected = {"a": value.to(dtype=torch.float)}
        self.assertDictOfTensorEqual(out, expected)
        self.assertTrue(out["a"].dtype == torch.float, msg="dtype != float")

        # happy path: value is list w/ elements type Tuple[Tensor, Tensor]
        presence = torch.tensor([[1, 1, 1], [1, 1, 1]])
        data = {
            "a": [
                (torch.tensor([[0, 0, 0], [1, 1, 1]]), presence),
                (torch.tensor([[2, 2, 2], [3, 3, 3]]), presence),
            ],
            "b": [
                (torch.tensor([[3, 3, 3], [2, 2, 2]]), presence),
                (torch.tensor([[1, 1, 1], [0, 0, 0]]), presence),
            ],
        }
        out = transforms.StackDenseFixedSizeArray(data.keys(), size=3)(data)
        expected = {
            "a": torch.tile(torch.arange(4).view(-1, 1).to(dtype=torch.float), (1, 3)),
            "b": torch.tile(
                torch.arange(4).flip(dims=(0,)).view(-1, 1).to(dtype=torch.float),
                (1, 3),
            ),
        }
        self.assertDictOfTensorEqual(out, expected)

        # raise for tensor wrong shape
        with self.assertRaisesRegex(ValueError, "Wrong shape"):
            sdf = transforms.StackDenseFixedSizeArray(["a"], size=3)
            sdf({"a": torch.ones(2)})

        # raise for tensor wrong ndim
        with self.assertRaisesRegex(ValueError, "Wrong shape"):
            sdf = transforms.StackDenseFixedSizeArray(["a"], size=2)
            sdf({"a": torch.zeros(2, 2, 2)})

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

    @patch("reagent.preprocessing.transforms.make_sparse_preprocessor")
    def test_MapIDListFeatures(self, mock_make_sparse_preprocessor):
        data = {
            InputColumn.STATE_ID_LIST_FEATURES: {0: [torch.tensor(1), torch.tensor(2)]},
            InputColumn.STATE_ID_SCORE_LIST_FEATURES: {
                1: [
                    torch.tensor(1),
                    torch.tensor(2),
                    torch.tensor(3),
                ]
            },
        }
        mock_make_sparse_preprocessor.return_value.preprocess_id_list.return_value = {
            InputColumn.STATE_ID_LIST_FEATURES: [torch.tensor(2), torch.tensor(3)]
        }
        mock_make_sparse_preprocessor.return_value.preprocess_id_score_list.return_value = {
            InputColumn.STATE_ID_SCORE_LIST_FEATURES: [
                torch.tensor(4),
                torch.tensor(5),
                torch.tensor(6),
            ]
        }
        state_id_list_columns: List[str] = [
            InputColumn.STATE_ID_LIST_FEATURES,
            InputColumn.NEXT_STATE_ID_LIST_FEATURES,
        ]
        state_id_score_list_columns: List[str] = [
            InputColumn.STATE_ID_SCORE_LIST_FEATURES,
            InputColumn.NEXT_STATE_ID_SCORE_LIST_FEATURES,
        ]
        state_feature_config = rlt.ModelFeatureConfig(
            id_list_feature_configs=[
                rlt.IdListFeatureConfig(
                    name=InputColumn.STATE_ID_LIST_FEATURES,
                    feature_id=0,
                    id_mapping_name="state_id_list_features_mapping",
                )
            ],
            id_score_list_feature_configs=[
                rlt.IdScoreListFeatureConfig(
                    name=InputColumn.STATE_ID_SCORE_LIST_FEATURES,
                    feature_id=1,
                    id_mapping_name="state_id_score_list_features_mapping",
                )
            ],
            id_mapping_config={
                "state_id_list_features_mapping": rlt.IdMappingUnion(
                    explicit_mapping=rlt.ExplicitMapping(ids=[0, 1, 2])
                ),
                "state_id_score_list_features_mapping": rlt.IdMappingUnion(
                    explicit_mapping=rlt.ExplicitMapping(ids=[3, 4, 5])
                ),
            },
        )

        map_id_list_features = transforms.MapIDListFeatures(
            id_list_keys=state_id_list_columns,
            id_score_list_keys=state_id_score_list_columns,
            feature_config=state_feature_config,
            device=torch.device("cpu"),
        )
        out = map_id_list_features(data)
        # output should contain all k in id_list_keys & id_score_list_keys
        self.assertEqual(len(out), 4)
        # The key should contain none if data don't have it
        self.assertIsNone(
            out[InputColumn.NEXT_STATE_ID_LIST_FEATURES], "It should be filtered out"
        )
        # The value of data changed based on sparse-preprocess mapping
        self.assertEqual(
            out[InputColumn.STATE_ID_LIST_FEATURES],
            {InputColumn.STATE_ID_LIST_FEATURES: [torch.tensor(2), torch.tensor(3)]},
        )
        # Testing assertion in the call method
        wrong_data = {
            InputColumn.STATE_ID_LIST_FEATURES: [torch.tensor(1), torch.tensor(2)],
            InputColumn.STATE_ID_SCORE_LIST_FEATURES: [
                torch.tensor(1),
                torch.tensor(2),
                torch.tensor(3),
            ],
        }
        with self.assertRaises(AssertionError):
            map_id_list_features(wrong_data)
        # Testing assertion in the constructor
        state_id_list_columns: List[str] = [
            InputColumn.STATE_ID_LIST_FEATURES,
            InputColumn.NEXT_STATE_ID_LIST_FEATURES,
        ]
        state_id_score_list_columns: List[str] = [
            InputColumn.STATE_ID_LIST_FEATURES,
            InputColumn.NEXT_STATE_ID_LIST_FEATURES,
        ]
        with self.assertRaises(AssertionError):
            transforms.MapIDListFeatures(
                id_list_keys=state_id_list_columns,
                id_score_list_keys=state_id_score_list_columns,
                feature_config=state_feature_config,
                device=torch.device("cpu"),
            )

    def test_FixedLengthSequences(self):
        # of form {sequence_id: (offsets, Tuple(Tensor, Tensor))}
        a_T = (torch.tensor([0, 1]), torch.tensor([1, 0]))
        b_T = (torch.tensor([1, 1]), torch.tensor([1, 0]))
        a_in = {1: (torch.tensor([0]), a_T)}
        b_in = {1: (torch.tensor([0, 2]), b_T)}
        fls1 = transforms.FixedLengthSequences(keys=["a", "b"], sequence_id=1)
        fls2 = transforms.FixedLengthSequences(
            keys=["a", "b"], sequence_id=1, expected_length=2
        )
        fls3 = transforms.FixedLengthSequences(
            keys=["a", "b"], sequence_id=1, expected_length=2, to_keys=["to_a", "to_b"]
        )
        o1 = fls1({"a": a_in, "b": b_in})
        o2 = fls2({"a": a_in, "b": b_in})
        o3 = fls3({"a": a_in, "b": b_in})
        # o1, o2 should contain only keys
        self.assertEqual(len(o1), 2)
        self.assertEqual(len(o2), 2)
        # o3 should contain keys & to_keys
        self.assertEqual(len(o3), 4)
        # ensure `T` is set back to key
        self.assertTrue(
            torch.all(o1["a"][0] == a_T[0]) and torch.all(o1["a"][1] == a_T[1])
        )
        self.assertTrue(
            torch.all(o1["b"][0] == b_T[0]) and torch.all(o1["b"][1] == b_T[1])
        )
        self.assertTrue(
            torch.all(o2["a"][0] == a_T[0]) and torch.all(o2["a"][1] == a_T[1])
        )
        self.assertTrue(
            torch.all(o2["b"][0] == b_T[0]) and torch.all(o2["b"][1] == b_T[1])
        )
        # ensure keys not changed
        self.assertEqual(o3["a"], a_in)
        self.assertEqual(o3["b"], b_in)
        # # ensure `T` is set to_key
        self.assertTrue(
            torch.all(o3["to_a"][0] == a_T[0]) and torch.all(o3["to_a"][1] == a_T[1])
        )
        self.assertTrue(
            torch.all(o3["to_b"][0] == b_T[0]) and torch.all(o3["to_b"][1] == b_T[1])
        )
        # Testing assertions in the call method
        # TODO testing assert regarding offsets length compared to value
        c_T = (torch.tensor([0, 1]), torch.tensor([1, 1]))
        with self.assertRaisesRegex(Exception, "Unexpected offsets"):
            # wrong expected length
            fls = transforms.FixedLengthSequences(
                keys=["a", "b"], sequence_id=1, expected_length=1
            )
            fls({"a": a_in, "b": b_in})
        with self.assertRaisesRegex(Exception, "Unexpected offsets"):
            # wrong offsets
            c_in = {1: (torch.tensor([0, 1]), c_T)}
            fls = transforms.FixedLengthSequences(keys=["a", "b", "c"], sequence_id=1)
            fls({"a": a_in, "b": b_in, "c": c_in})
        # Testing assertion in the constructor
        with self.assertRaises(AssertionError):
            transforms.FixedLengthSequences(
                keys=["a", "b"], sequence_id=1, to_keys=["to_a"]
            )

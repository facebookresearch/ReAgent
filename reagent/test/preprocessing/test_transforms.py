import unittest
from copy import deepcopy
from typing import List
from unittest.mock import Mock, patch

import numpy as np
import reagent.core.types as rlt
import torch
from reagent.preprocessing import transforms
from reagent.preprocessing.types import InputColumn


class TestTransforms(unittest.TestCase):
    def setUp(self):
        # add custom compare function for torch.Tensor
        self.addTypeEqualityFunc(torch.Tensor, TestTransforms.are_torch_tensor_equal)

    @staticmethod
    def are_torch_tensor_equal(tensor_0, tensor_1, msg=None):
        if torch.all(tensor_0 == tensor_1):
            return True
        raise TestTransforms.failureException("non-equal pytorch tensors found", msg)

    def assertTorchTensorEqual(self, tensor_0, tensor_1, msg=None):
        self.assertIsInstance(
            tensor_0, torch.Tensor, "first argument is not a torch.Tensor"
        )
        self.assertIsInstance(
            tensor_1, torch.Tensor, "second argument is not a torch.Tensor"
        )
        self.assertEqual(tensor_0, tensor_1, msg=msg)

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

        self.assertEqual(torch.stack(in_1), torch.stack(a_in))
        self.assertEqual(torch.stack(in_2), torch.stack(b_in))

    @patch("reagent.preprocessing.transforms.Preprocessor")
    def test_FixedLengthSequenceDenseNormalization(self, Preprocessor):
        # test key mapping
        rand_gen = torch.Generator().manual_seed(0)

        a_batch_size = 2
        b_batch_size = 3

        a_dim = 13
        b_dim = 11

        expected_length = 7

        a_T = (
            torch.rand(
                a_batch_size * expected_length, a_dim, generator=rand_gen
            ),  # value
            torch.rand(a_batch_size * expected_length, a_dim, generator=rand_gen)
            > 0.5,  # presence
        )
        b_T = (
            torch.rand(
                b_batch_size * expected_length, b_dim, generator=rand_gen
            ),  # value
            torch.rand(b_batch_size * expected_length, b_dim, generator=rand_gen)
            > 0.5,  # presence
        )

        # expected values after preprocessing
        a_TN = a_T[0] + 1
        b_TN = b_T[0] + 1

        # copy used for checking inplace modifications
        a_TN_copy = deepcopy(a_TN)
        b_TN_copy = deepcopy(b_TN)

        a_offsets = torch.arange(0, a_batch_size * expected_length, expected_length)
        b_offsets = torch.arange(0, b_batch_size * expected_length, expected_length)

        a_in = {1: (a_offsets, a_T), 2: 0}
        b_in = {1: (b_offsets, b_T), 2: 1}

        c_out = 2

        # input data
        data = {"a": a_in, "b": b_in, "c": c_out}

        # copy used for checking inplace modifications
        data_copy = deepcopy(data)

        Preprocessor.return_value = Mock(side_effect=[a_TN, b_TN])

        flsdn = transforms.FixedLengthSequenceDenseNormalization(
            keys=["a", "b"],
            sequence_id=1,
            normalization_data=Mock(),
        )

        out = flsdn(data)

        # data is modified inplace and returned
        self.assertEqual(data, out)

        # check preprocessor number of calls
        self.assertEqual(Preprocessor.call_count, 1)
        self.assertEqual(Preprocessor.return_value.call_count, 2)

        # result contains original keys and new processed keys
        self.assertSetEqual(set(out.keys()), {"a", "b", "c", "a:1", "b:1"})

        def assertKeySeqIdItem(item_0, item_1):
            self.assertTorchTensorEqual(item_0[0], item_1[0])
            self.assertTorchTensorEqual(item_0[1][0], item_1[1][0])
            self.assertTorchTensorEqual(item_0[1][1], item_1[1][1])

        # original keys should keep their value
        for key in ("a", "b"):
            # no change in the output
            assertKeySeqIdItem(out[key][1], data_copy[key][1])

            # no change in untouched seq id
            self.assertEqual(out[key][2], data_copy[key][2])

        # no change in the non-processed key
        self.assertEqual(out["c"], data_copy["c"])

        # check output shapes
        self.assertListEqual(
            [*out["a:1"].shape], [a_batch_size, expected_length, a_dim]
        )
        self.assertListEqual(
            [*out["b:1"].shape], [b_batch_size, expected_length, b_dim]
        )

        # no inplace change in normalized tensors
        self.assertTorchTensorEqual(a_TN, a_TN_copy)
        self.assertTorchTensorEqual(b_TN, b_TN_copy)

        # check if output has been properly slated
        self.assertTorchTensorEqual(
            out["a:1"], a_TN.view(a_batch_size, expected_length, a_dim)
        )
        self.assertTorchTensorEqual(
            out["b:1"], b_TN.view(b_batch_size, expected_length, b_dim)
        )

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

    def test_OneHotActions(self):
        keys = ["0", "1", "2"]
        num_actions = 2
        oha = transforms.OneHotActions(keys, num_actions)
        data_in = {"0": torch.tensor(0), "1": torch.tensor(1), "2": torch.tensor(2)}
        data_out = oha(data_in)
        expected = {
            "0": torch.tensor([1, 0]),
            "1": torch.tensor([0, 1]),
            "2": torch.tensor([0, 0]),
        }
        self.assertDictOfTensorEqual(data_out, expected)

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

    def test_SlateView(self):
        # Unit tests for the SlateView class
        sv = transforms.SlateView(keys=["a"], slate_size=-1)

        # GIVEN a SlateView with keys = ["a"]
        # WHEN data is passed in under a key "b"
        # THEN the value for "b" should not be unflattened since the key "b" is not in SlateView.keys!
        sv.slate_size = 1
        sv.keys = ["a"]
        a_in = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
        b_in = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
        data = {"a": a_in, "b": b_in}
        out = sv(data)
        self.assertEqual(out["b"].shape, torch.Size([4, 2]))
        self.assertTorchTensorEqual(out["b"], b_in)

        # GIVEN slate.size = 1 and keys = ["a", "b"]
        # WHEN input shape is [4, 2]
        # THEN output shape should be [4, 1, 2] for all keys
        sv.slate_size = 1
        sv.keys = ["a", "b"]
        a_in = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
        b_in = torch.tensor([[10, 20], [30, 40], [50, 60], [70, 80]])
        data = {"a": a_in, "b": b_in}
        out = sv(data)
        a_out_412 = torch.tensor([[[1, 2]], [[3, 4]], [[5, 6]], [[7, 8]]])
        b_out_412 = torch.tensor([[[10, 20]], [[30, 40]], [[50, 60]], [[70, 80]]])
        self.assertEqual(out["a"].shape, torch.Size([4, 1, 2]))
        self.assertEqual(out["b"].shape, torch.Size([4, 1, 2]))
        self.assertDictOfTensorEqual({"a": a_out_412, "b": b_out_412}, out)

        # GIVEN a SlateView with keys = ["a", "b"]
        # WHEN data is passed in missing one or more of those keys
        # THEN a KeyError should be raised
        sv.keys = ["a", "b"]
        a_in = torch.tensor([[1, 2], [3, 4]])
        c_in = torch.tensor([[1, 2], [3, 4]])
        data = {"a": a_in, "c": c_in}
        with self.assertRaises(KeyError):
            out = sv(data)

        # GIVEN a SlateView with keys = ["a"]
        # WHEN data is passed in that is of an invalid shape
        # THEN a RuntimeError should be raised
        sv.slate_size = 2
        sv.keys = ["a"]
        a_in = torch.tensor([[1, 2]])
        data = {"a": a_in}
        with self.assertRaises(RuntimeError):
            out = sv(data)

        # GIVEN slate.size = 2 and keys = ["a"]
        # WHEN input shape is [4, 3]
        # THEN output shape should be [2, 2, 3]
        sv.slate_size = 2
        sv.keys = ["a"]
        a_in = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        data = {"a": a_in}
        out = sv(data)
        a_out_223 = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
        self.assertEqual(out["a"].shape, torch.Size([2, 2, 3]))
        self.assertDictOfTensorEqual({"a": a_out_223}, out)

    def _check_same_keys(self, dict_a, dict_b):
        self.assertSetEqual(set(dict_a.keys()), set(dict_b.keys()))

    def test_AppendConstant(self):
        data = {
            "a": torch.tensor([[9.0, 4.5], [3.4, 3.9]]),
            "b": torch.tensor([[9.2, 2.5], [4.4, 1.9]]),
        }
        t = transforms.AppendConstant(["a"], const=1.5)
        t_data = t(data)
        self._check_same_keys(data, t_data)
        self.assertTorchTensorEqual(data["b"], t_data["b"])
        self.assertTorchTensorEqual(
            t_data["a"], torch.tensor([[1.5, 9.0, 4.5], [1.5, 3.4, 3.9]])
        )

    def test_UnsqueezeRepeat(self):
        data = {
            "a": torch.tensor([[9.0, 4.5], [3.4, 3.9]]),
            "b": torch.tensor([[9.2, 2.5], [4.4, 1.9]]),
        }
        t = transforms.UnsqueezeRepeat(["a"], dim=1, num_repeat=3)
        t_data = t(data)
        self._check_same_keys(data, t_data)
        self.assertTorchTensorEqual(data["b"], t_data["b"])
        self.assertTorchTensorEqual(
            t_data["a"],
            torch.tensor(
                [
                    [[9.0, 4.5], [9.0, 4.5], [9.0, 4.5]],
                    [[3.4, 3.9], [3.4, 3.9], [3.4, 3.9]],
                ]
            ),
        )

    def test_OuterProduct(self):
        data = {
            "a": torch.tensor([[9.0, 4.5], [3.4, 3.9]]),
            "b": torch.tensor([[9.2, 2.5], [4.4, 1.9]]),
        }
        t = transforms.OuterProduct("a", "b", "ab")
        t_data = t(data)
        # make sure original data was left unmodified
        self.assertTorchTensorEqual(data["a"], t_data["a"])
        self.assertTorchTensorEqual(data["b"], t_data["b"])

        expected_out = torch.empty(2, 4)
        for i in range(2):
            expected_out[i, :] = torch.outer(
                data["a"][i, :].flatten(), data["b"][i, :].flatten()
            ).flatten()
        self.assertTorchTensorEqual(t_data["ab"], expected_out)

    def test_GetEye(self):
        data = {
            "a": torch.tensor([[9.0, 4.5], [3.4, 3.9]]),
            "b": torch.tensor([[9.2, 2.5], [4.4, 1.9]]),
        }
        t = transforms.GetEye("c", 4)
        t_data = t(data)
        # make sure original data was left unmodified
        self.assertTorchTensorEqual(data["a"], t_data["a"])
        self.assertTorchTensorEqual(data["b"], t_data["b"])

        self.assertTorchTensorEqual(t_data["c"], torch.eye(4))

    def test_Cat(self):
        data = {
            "a": torch.tensor([[9.0, 4.5], [3.4, 3.9]]),
            "b": torch.tensor([[9.2, 2.5], [4.4, 1.9]]),
        }
        t = transforms.Cat(["a", "b"], "c", 0)
        t_data = t(data)
        # make sure original data was left unmodified
        self.assertTorchTensorEqual(data["a"], t_data["a"])
        self.assertTorchTensorEqual(data["b"], t_data["b"])

        self.assertTorchTensorEqual(t_data["c"], torch.cat([data["a"], data["b"]], 0))

    def test_Rename(self):
        data = {
            "a": torch.tensor([[9.0, 4.5], [3.4, 3.9]]),
            "b": torch.tensor([[9.2, 2.5], [4.4, 1.9]]),
        }
        t = transforms.Rename(["a"], ["aa"])
        t_data = t(data)
        # make sure original data was left unmodified
        self.assertTorchTensorEqual(data["b"], t_data["b"])

        self.assertTorchTensorEqual(t_data["aa"], data["a"])

    def test_Filter(self):
        data = {
            "a": torch.tensor([[9.0, 4.5], [3.4, 3.9]]),
            "b": torch.tensor([[9.2, 2.5], [4.4, 1.9]]),
        }
        t = transforms.Filter(keep_keys=["a"])
        t_data = t(data)
        # make sure original data was left unmodified
        self.assertTorchTensorEqual(data["a"], t_data["a"])
        self.assertListEqual(sorted(t_data.keys()), ["a"])

        t = transforms.Filter(remove_keys=["b"])
        t_data = t(data)
        # make sure original data was left unmodified
        self.assertTorchTensorEqual(data["a"], t_data["a"])
        self.assertListEqual(sorted(t_data.keys()), ["a"])

    def test_broadcast_tensors_for_cat(self):
        tensors = [
            torch.tensor([[3.0, 4.0, 5.0], [4.5, 4.3, 5.9]]),
            torch.tensor([[2.0, 9.0, 8.0]]),
        ]
        broadcasted_tensors = transforms._broadcast_tensors_for_cat(tensors, 1)
        self.assertTorchTensorEqual(broadcasted_tensors[0], tensors[0])
        self.assertTorchTensorEqual(broadcasted_tensors[1], tensors[1].repeat(2, 1))

        tensors = [
            torch.empty(10, 2, 5),
            torch.empty(1, 2, 3),
        ]
        broadcasted_tensors = transforms._broadcast_tensors_for_cat(tensors, -1)
        self.assertEqual(tuple(broadcasted_tensors[0].shape), (10, 2, 5))
        self.assertEqual(tuple(broadcasted_tensors[1].shape), (10, 2, 3))

        tensors = [
            torch.empty(1, 1, 5),
            torch.empty(10, 3, 1),
        ]
        broadcasted_tensors = transforms._broadcast_tensors_for_cat(tensors, 1)
        self.assertEqual(tuple(broadcasted_tensors[0].shape), (10, 1, 5))
        self.assertEqual(tuple(broadcasted_tensors[1].shape), (10, 3, 5))

        tensors = [
            torch.empty(1, 3, 5, 1),
            torch.empty(10, 3, 1, 4),
        ]
        broadcasted_tensors = transforms._broadcast_tensors_for_cat(tensors, 0)
        self.assertEqual(tuple(broadcasted_tensors[0].shape), (1, 3, 5, 4))
        self.assertEqual(tuple(broadcasted_tensors[1].shape), (10, 3, 5, 4))

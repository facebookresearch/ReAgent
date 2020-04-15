#!/usr/bin/env python3

import unittest
from typing import Tuple, Union

import numpy as np
import torch
from reagent.ope.estimators.types import Distribution, Items, TypeWrapper, Values
from torch import Tensor


class TestTypes(unittest.TestCase):
    TestType = Union[int, Tuple[int], float, Tuple[float], np.ndarray, Tensor]
    TestClass = TypeWrapper[TestType]

    def setUp(self) -> None:
        self._test_list = [0, 1, 2, 3, 5]

    def test_int_type(self):
        int_val = TestTypes.TestClass(3)
        self.assertEqual(self._test_list[int_val], 3)
        self.assertEqual(hash(int_val), hash(3))
        int_val_other = TestTypes.TestClass(3)
        self.assertEqual(int_val, int_val_other)
        int_val_other = TestTypes.TestClass(4)
        self.assertNotEqual(int_val, int_val_other)

    def test_float_type(self):
        float_val = TestTypes.TestClass(3.2)
        self.assertEqual(self._test_list[float_val], 3)
        self.assertEqual(hash(float_val), hash(3.2))
        float_val_other = TestTypes.TestClass(3.2)
        self.assertEqual(float_val, float_val_other)
        float_val_other = TestTypes.TestClass(4.3)
        self.assertNotEqual(float_val, float_val_other)

    def test_tuple_int_type(self):
        tuple_int_val = TestTypes.TestClass((1, 2, 3))
        with self.assertRaises(ValueError):
            self._test_list[tuple_int_val] = 1
        self.assertEqual(hash(tuple_int_val), hash((1, 2, 3)))
        tuple_int_val_other = TestTypes.TestClass((1, 2, 3))
        self.assertEqual(tuple_int_val, tuple_int_val_other)
        tuple_int_val_other = TestTypes.TestClass((2, 3, 1))
        self.assertNotEqual(tuple_int_val, tuple_int_val_other)

    def test_tuple_float_type(self):
        tuple_float_val = TestTypes.TestClass((1.1, 2.2, 3.3))
        with self.assertRaises(ValueError):
            self._test_list[tuple_float_val] = 1
        self.assertEqual(hash(tuple_float_val), hash((1.1, 2.2, 3.3)))
        tuple_float_val_other = TestTypes.TestClass((1.1, 2.2, 3.3))
        self.assertEqual(tuple_float_val, tuple_float_val_other)
        tuple_float_val_other = TestTypes.TestClass((2.2, 3.3, 1.1))
        self.assertNotEqual(tuple_float_val, tuple_float_val_other)

    def test_ndarray_type(self):
        ndarray_val = TestTypes.TestClass(np.array(3))
        self.assertEqual(self._test_list[ndarray_val], 3)
        self.assertEqual(hash(ndarray_val), hash((3,)))
        ndarray_val_other = TestTypes.TestClass(np.array(3))
        self.assertEqual(ndarray_val, ndarray_val_other)
        int_val_other = TestTypes.TestClass(3)
        self.assertEqual(ndarray_val, int_val_other)
        ndarray_val_other = TestTypes.TestClass(np.array(4))
        self.assertNotEqual(ndarray_val, ndarray_val_other)
        ndarray_val = TestTypes.TestClass(np.array(((1, 2), (3, 4))))
        with self.assertRaises(ValueError):
            self._test_list[ndarray_val] = 1
        self.assertEqual(hash(ndarray_val), hash((1, 2, 3, 4)))
        ndarray_val_other = TestTypes.TestClass(((1, 2), (3, 4)))
        self.assertEqual(ndarray_val, ndarray_val_other)
        ndarray_val_other = TestTypes.TestClass(np.ndarray((1, 2, 3, 4)))
        self.assertNotEqual(ndarray_val, ndarray_val_other)

    def test_tensor_type(self):
        tensor_val = TestTypes.TestClass(torch.tensor(3))
        self.assertEqual(self._test_list[tensor_val], 3)
        self.assertEqual(hash(tensor_val), hash((3,)))
        tensor_val_other = TestTypes.TestClass(torch.tensor(3))
        self.assertEqual(tensor_val, tensor_val_other)
        int_val_other = TestTypes.TestClass(3)
        with self.assertRaises(TypeError):
            _ = tensor_val == int_val_other
        tensor_val_other = TestTypes.TestClass(torch.tensor(4))
        self.assertNotEqual(tensor_val, tensor_val_other)
        tensor_val = TestTypes.TestClass(torch.tensor(((1, 2), (3, 4))))
        with self.assertRaises(ValueError):
            self._test_list[tensor_val] = 1
        self.assertEqual(hash(tensor_val), hash((1, 2, 3, 4)))
        tensor_val_other = TestTypes.TestClass(torch.tensor((1, 2, 3, 4)))
        self.assertNotEqual(tensor_val, tensor_val_other)


class TestValues(unittest.TestCase):
    TestIntType = TypeWrapper[int]
    TestTupleFloatType = TypeWrapper[Tuple[float]]

    class TestIntKeyValues(Values[TestIntType]):
        def _new_key(self, k: int):
            return TestValues.TestIntType(k)

    class TestTupleFloatKeyValues(Values[TestTupleFloatType]):
        def _new_key(self, k: int):
            raise TypeError(
                f"value {k} invalid for " f"{TestValues.TestTupleFloatType.__name__}"
            )

    def setUp(self) -> None:
        self._int_float_values = TestValues.TestIntKeyValues([2.2, 4.4, 1.1, 3.3])
        self._tuple_float_float_values = TestValues.TestTupleFloatKeyValues(
            {
                TestValues.TestTupleFloatType((1.0, 2.0)): 2.2,
                TestValues.TestTupleFloatType((3.0, 4.0)): 4.4,
                TestValues.TestTupleFloatType((5.0, 6.0)): 1.1,
                TestValues.TestTupleFloatType((7.0, 8.0)): 3.3,
            }
        )
        self._int_array_values = TestValues.TestIntKeyValues(
            np.array((2.2, 4.4, 1.1, 3.3))
        )
        self._int_tensor_values = TestValues.TestIntKeyValues(
            torch.tensor((2.2, 4.4, 1.1, 3.3))
        )

    def test_indexing(self):
        self.assertEqual(self._int_float_values[2], 1.1)
        self.assertEqual(self._int_float_values[TestValues.TestIntType(2)], 1.1)
        self.assertEqual(
            self._tuple_float_float_values[TestValues.TestTupleFloatType((3.0, 4.0))],
            4.4,
        )

    def test_sort(self):
        keys, values = self._int_float_values.sort()
        self.assertEqual(
            keys,
            [
                TestValues.TestIntType(1),
                TestValues.TestIntType(3),
                TestValues.TestIntType(0),
                TestValues.TestIntType(2),
            ],
        )
        self.assertEqual(values, [4.4, 3.3, 2.2, 1.1])
        keys, values = self._tuple_float_float_values.sort()
        self.assertEqual(
            keys,
            [
                TestValues.TestTupleFloatType((3.0, 4.0)),
                TestValues.TestTupleFloatType((7.0, 8.0)),
                TestValues.TestTupleFloatType((1.0, 2.0)),
                TestValues.TestTupleFloatType((5.0, 6.0)),
            ],
        )
        self.assertEqual(values, [4.4, 3.3, 2.2, 1.1])
        keys, values = self._int_array_values.sort()
        self.assertEqual(
            keys,
            [
                TestValues.TestIntType(1),
                TestValues.TestIntType(3),
                TestValues.TestIntType(0),
                TestValues.TestIntType(2),
            ],
        )
        self.assertTrue(np.array_equal(values, np.array([4.4, 3.3, 2.2, 1.1])))
        keys, values = self._int_tensor_values.sort()
        self.assertEqual(
            keys,
            [
                TestValues.TestIntType(1),
                TestValues.TestIntType(3),
                TestValues.TestIntType(0),
                TestValues.TestIntType(2),
            ],
        )
        self.assertTrue(torch.equal(values, torch.tensor([4.4, 3.3, 2.2, 1.1])))

    def test_unzip(self):
        items = self._int_float_values.items
        values = self._int_float_values.values
        self.assertEqual(
            items,
            [
                TestValues.TestIntType(0),
                TestValues.TestIntType(1),
                TestValues.TestIntType(2),
                TestValues.TestIntType(3),
            ],
        )
        self.assertEqual(values, [2.2, 4.4, 1.1, 3.3])
        items = self._tuple_float_float_values.items
        values = self._tuple_float_float_values.values
        self.assertEqual(
            items,
            [
                TestValues.TestTupleFloatType((1.0, 2.0)),
                TestValues.TestTupleFloatType((3.0, 4.0)),
                TestValues.TestTupleFloatType((5.0, 6.0)),
                TestValues.TestTupleFloatType((7.0, 8.0)),
            ],
        )
        self.assertEqual(values, [2.2, 4.4, 1.1, 3.3])
        items = self._int_array_values.items
        values = self._int_array_values.values
        self.assertEqual(
            items,
            [
                TestValues.TestIntType(0),
                TestValues.TestIntType(1),
                TestValues.TestIntType(2),
                TestValues.TestIntType(3),
            ],
        )
        self.assertTrue(np.array_equal(values, np.array([2.2, 4.4, 1.1, 3.3])))
        items = self._int_tensor_values.items
        values = self._int_tensor_values.values
        self.assertEqual(
            items,
            [
                TestValues.TestIntType(0),
                TestValues.TestIntType(1),
                TestValues.TestIntType(2),
                TestValues.TestIntType(3),
            ],
        )
        self.assertTrue(torch.equal(values, torch.tensor([2.2, 4.4, 1.1, 3.3])))

    def test_copy(self):
        copy = self._int_float_values.copy()
        for i, c in zip(self._int_float_values, copy):
            self.assertEqual(i, c)
        copy[1] = 2.1
        self.assertNotEqual(copy[1], self._int_float_values[1])
        copy = self._tuple_float_float_values.copy()
        for i, c in zip(self._tuple_float_float_values, copy):
            self.assertEqual(i, c)
        key = TestValues.TestTupleFloatType((3.0, 4.0))
        copy[key] = 2.1
        self.assertNotEqual(copy[key], self._tuple_float_float_values[key])
        copy = self._int_array_values.copy()
        for i, c in zip(self._int_array_values, copy):
            self.assertEqual(i, c)
        copy[1] = 2.1
        self.assertNotEqual(copy[1], self._int_array_values[1])
        copy = self._int_tensor_values.copy()
        for i, c in zip(self._int_tensor_values, copy):
            self.assertEqual(i, c)
        copy[1] = 2.1
        self.assertNotEqual(copy[1], self._int_tensor_values[1])

    def test_conversion(self):
        float_list_val = [1.1, 2.2, 3.3]
        tensor_val = torch.tensor([1.1, 2.2, 3.3], dtype=torch.double)
        array_val = np.array([1.1, 2.2, 3.3], dtype=np.float64)
        self.assertTrue(
            torch.equal(
                Values.to_tensor(float_list_val, dtype=torch.double), tensor_val
            )
        )
        self.assertTrue(
            torch.equal(Values.to_tensor(tensor_val, dtype=torch.double), tensor_val)
        )
        self.assertTrue(
            torch.equal(Values.to_tensor(array_val, dtype=torch.double), tensor_val)
        )
        self.assertTrue(np.array_equal(Values.to_ndarray(float_list_val), array_val))
        self.assertTrue(
            np.array_equal(Values.to_ndarray(tensor_val, dtype=np.float64), array_val)
        )
        self.assertTrue(np.array_equal(Values.to_ndarray(array_val), array_val))
        self.assertEqual(Values.to_sequence(float_list_val), float_list_val)
        self.assertEqual(Values.to_sequence(tensor_val), float_list_val)
        self.assertEqual(Values.to_sequence(array_val), float_list_val)


class TestDistribution(unittest.TestCase):
    class TestIntKeyDistribution(Distribution[int]):
        def _new_key(self, k: int):
            return k

    def setUp(self) -> None:
        self._tensor_distribution = TestDistribution.TestIntKeyDistribution(
            torch.tensor([1.0, 2.0, 3.0, 4.0])
        )
        self._array_distribution = TestDistribution.TestIntKeyDistribution(
            np.array([1.0, 2.0, 3.0, 4.0])
        )
        self._list_distribution = TestDistribution.TestIntKeyDistribution(
            [1.0, 2.0, 3.0, 4.0]
        )
        self._map_distribution = TestDistribution.TestIntKeyDistribution(
            {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}
        )

    def test_values(self):
        self.assertTrue(
            torch.equal(
                self._tensor_distribution.values, torch.tensor([0.1, 0.2, 0.3, 0.4])
            )
        )
        self.assertTrue(
            np.array_equal(
                self._array_distribution.values, np.array([0.1, 0.2, 0.3, 0.4])
            )
        )
        self.assertEqual(self._list_distribution.values, [0.1, 0.2, 0.3, 0.4])
        self.assertTrue(self._map_distribution.values, [0.1, 0.2, 0.3, 0.4])

    def _test_sample(self, distribution: Distribution):
        counts = [0] * 4
        total = 100000
        for _ in range(total):
            counts[distribution.sample()] += 1
        self.assertAlmostEqual(counts[0] / total, 0.1, places=2)
        self.assertAlmostEqual(counts[1] / total, 0.2, places=2)
        self.assertAlmostEqual(counts[2] / total, 0.3, places=2)
        self.assertAlmostEqual(counts[3] / total, 0.4, places=2)

    def test_sample(self):
        self._test_sample(self._tensor_distribution)
        self.assertEqual(self._tensor_distribution.greedy(4), [3, 2, 1, 0])
        self._test_sample(self._array_distribution)
        self.assertEqual(self._array_distribution.greedy(4), [3, 2, 1, 0])
        self._test_sample(self._list_distribution)
        self.assertEqual(self._list_distribution.greedy(4), [3, 2, 1, 0])
        self._test_sample(self._map_distribution)
        self.assertEqual(self._map_distribution.greedy(4), [3, 2, 1, 0])


if __name__ == "__main__":
    np.random.seed(1234)
    torch.random.manual_seed(1234)

    unittest.main()

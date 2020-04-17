#!/usr/bin/env python3

import unittest

import numpy as np
import torch
from reagent.ope.utils import Clamper, RunningAverage


class TestUtils(unittest.TestCase):
    def test_running_average(self):
        ra = RunningAverage()
        ra.add(1.0).add(2.0).add(3.0).add(4.0)
        self.assertEqual(ra.count, 4)
        self.assertEqual(ra.average, 2.5)
        self.assertEqual(ra.total, 10.0)

    def test_clamper(self):
        with self.assertRaises(ValueError):
            clamper = Clamper(1.0, 0.0)
        list_value = [-1.1, 0.9, 0.0, 1.1, -0.9]
        tensor_value = torch.tensor(list_value)
        array_value = np.array(list_value)
        clamper = Clamper()
        self.assertEqual(clamper(list_value), list_value)
        self.assertTrue(torch.equal(clamper(tensor_value), tensor_value))
        self.assertTrue(np.array_equal(clamper(array_value), array_value))
        clamper = Clamper(-1.0, 1.0)
        self.assertEqual(clamper(list_value), [-1.0, 0.9, 0.0, 1.0, -0.9])
        self.assertTrue(
            torch.equal(
                clamper(tensor_value), torch.tensor([-1.0, 0.9, 0.0, 1.0, -0.9])
            )
        )
        self.assertTrue(
            np.array_equal(clamper(array_value), np.array([-1.0, 0.9, 0.0, 1.0, -0.9]))
        )


if __name__ == "__main__":
    unittest.main()

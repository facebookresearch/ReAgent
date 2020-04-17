#!/usr/bin/env python3

import random
import unittest
from functools import reduce

import torch
from reagent.ope.estimators.slate_estimators import (
    DCGSlateMetric,
    NDCGSlateMetric,
    Slate,
    SlateItem,
    SlateItemProbabilities,
    SlateItemValues,
    SlateSlotItemProbabilities,
    SlateSlots,
)


class TestEstimator(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(1234)
        torch.random.manual_seed(1234)
        self._item_relevances = [1.0, 0.5, 2.5, 2.0, 5.0]
        self._slot_item_relevances = [
            [1.0, 0.5, 2.5, 2.0, 5.0],
            [1.5, 1.0, 2.0, 1.0, 4.0],
            [3.0, 2.5, 0.5, 4.0, 2.0],
        ]
        self._item_rewards = [3.0, 8.0, 0.0, 4.0, 1.0]
        self._item_size = len(self._item_relevances)
        self._slate_size = 3
        self._slots = SlateSlots(self._slate_size)

    def test_slate_item_probabilities(self):
        probs = SlateItemProbabilities(self._item_relevances)
        slate = probs.sample_slate(self._slots)
        slate_prob = probs.slate_probability(slate)
        self.assertAlmostEqual(slate_prob, 0.017825312)
        slot_item_expectations = probs.slot_item_expectations(self._slots)
        slot_rewards = slot_item_expectations.expected_rewards(
            SlateItemValues(self._item_rewards)
        )
        diff = slot_rewards.values - torch.tensor([1.81818, 2.13736, 2.66197])
        self.assertAlmostEqual(diff.sum().item(), 0, places=5)

    def test_slate_slot_item_probabilities(self):
        probs = SlateSlotItemProbabilities(
            [SlateItemValues(vs) for vs in self._slot_item_relevances]
        )
        slate = probs.sample_slate(self._slots)
        slate_prob = probs.slate_probability(slate)
        self.assertAlmostEqual(slate_prob, 0.02139037)
        slot_item_expectations = probs.slot_item_expectations()
        slot_rewards = slot_item_expectations.expected_rewards(
            SlateItemValues(self._item_rewards)
        )
        diff = slot_rewards.values - torch.tensor([1.818, 2.449, 4.353])
        self.assertAlmostEqual(diff.sum().item(), 0, places=5)
        for d in slot_item_expectations.items:
            sum = reduce(lambda a, b: a + b, d.values)
            self.assertAlmostEqual(sum.item(), 1.0)

    def test_metrics(self):
        dcg = DCGSlateMetric()
        ndcg = NDCGSlateMetric(SlateItemValues([1.0, 2.5, 2.0, 3.0, 1.5, 0.0]))
        item_rewards = SlateItemValues([2.0, 1.0, 0.0, 3.0, 1.5, 2.5])
        slate = Slate([SlateItem(1), SlateItem(3), SlateItem(2)])
        reward = dcg(slate.slots, slate.slot_values(item_rewards))
        self.assertAlmostEqual(reward, 5.416508275)
        reward = ndcg(slate.slots, slate.slot_values(item_rewards))
        self.assertAlmostEqual(reward, 0.473547669)
        slate = Slate([SlateItem(5), SlateItem(0), SlateItem(4)])
        reward = dcg(slate.slots, slate.slot_values(item_rewards))
        self.assertAlmostEqual(reward, 7.463857073)
        reward = ndcg(slate.slots, slate.slot_values(item_rewards))
        self.assertAlmostEqual(reward, 0.652540703)

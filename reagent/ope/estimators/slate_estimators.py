#!/usr/bin/env python3

import logging
import math
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Iterable,
    Mapping,
    MutableMapping,
    MutableSequence,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
import torch
from reagent.ope.estimators.estimator import (
    Estimator,
    EstimatorResult,
    EstimatorSampleResult,
)
from reagent.ope.estimators.types import (
    Action,
    Items,
    Objects,
    Probability,
    Reward,
    Trainer,
    TrainingData,
    TypeWrapper,
    Values,
    ValueType,
)
from reagent.ope.utils import Clamper, RunningAverage
from torch import Tensor


# Main algorithms are from two papers:
#   1. Offline Evaluation of Ranking Policies with Click Models
#      https://arxiv.org/abs/1804.10488
#   2. Off-policy evaluation for slate recommendation
#      https://arxiv.org/abs/1605.04812

# Types for slates
SlateSlotType = Union[int, Tuple[int], float, Tuple[float], np.ndarray, Tensor]
SlateSlot = TypeWrapper[SlateSlotType]
logger = logging.getLogger(__name__)


class SlateSlotValues(Values[SlateSlot]):
    """
    Map from a slot to a value
    """

    def _to_key(self, k: int) -> SlateSlot:
        return SlateSlot(k)


class SlateSlots(Items[SlateSlot]):
    """
    List of slot
    """

    def _new_item(self, i: int) -> SlateSlot:
        return SlateSlot(i)

    # pyre-fixme[15]: `fill` overrides method defined in `Items` inconsistently.
    def fill(
        self,
        values: Union[Mapping[SlateSlot, float], Sequence[float], np.ndarray, Tensor],
    ) -> SlateSlotValues:
        """
        Map slots to given values
        Args:
            values: given values

        Returns:
            Map from slots to given values
        """
        return SlateSlotValues(super().fill(values))


class SlateSlotObjects(Objects[SlateSlot, ValueType]):
    def __init__(
        self,
        values: Union[MutableMapping[SlateSlot, ValueType], MutableSequence[ValueType]],
    ):
        assert (len(values)) > 0
        super().__init__(values)

    def _to_key(self, k: int) -> SlateSlot:
        return SlateSlot(k)

    @property
    def slots(self) -> SlateSlots:
        if self.is_sequence:
            # pyre-fixme[16]: `SlateSlotObjects` has no attribute `_values`.
            return SlateSlots(len(self._values))
        else:
            return SlateSlots(list(self._key_to_index.keys()))

    @property
    def objects(self) -> Sequence[ValueType]:
        return super().values

    def fill(
        self, values: Sequence[ValueType]
    ) -> Union[Mapping[SlateSlot, ValueType], Sequence[ValueType]]:
        # pyre-fixme[16]: `SlateSlotObjects` has no attribute `_values`.
        assert len(values) >= len(self._values)
        if self._key_to_index is None:
            return values[: len(self._values)]
        else:
            return {s: v for s, v in zip(self.slots, values[: len(self._values)])}


# type of valid slate candidates, e.g., doc id
SlateItem = Action


class SlateItems(Items[SlateItem]):
    def _new_item(self, i: int) -> SlateItem:
        return SlateItem(i)


class SlateItemValues(Values[SlateItem]):
    def _to_key(self, k: int) -> SlateItem:
        return SlateItem(k)

    @property
    def items(self) -> SlateItems:
        if self.is_sequence:
            return SlateItems(len(self))
        else:
            return SlateItems(super().keys)


class SlateItemFeatures(Objects[SlateItem, Tensor]):
    def __init__(
        self,
        values: Union[Mapping[SlateItem, Tensor], Sequence[Tensor], Tensor, np.ndarray],
    ):
        # pyre-fixme[6]: Expected
        #  `Union[Mapping[Variable[reagent.ope.estimators.types.KeyType],
        #  Variable[ValueType]], Sequence[Variable[ValueType]]]` for 1st param but got
        #  `Union[Mapping[TypeWrapper[Union[Tuple[float], Tuple[int], Tensor, float,
        #  int, np.ndarray]], Tensor], Sequence[Tensor], Tensor, np.ndarray]`.
        super().__init__(values)

    def _init_values(
        self,
        values: Union[Mapping[SlateItem, Tensor], Sequence[Tensor], Tensor, np.ndarray],
    ):
        if isinstance(values, Tensor):
            # pyre-fixme[16]: `SlateItemFeatures` has no attribute `_values`.
            self._values = values.to(dtype=torch.double)
        elif isinstance(values, np.ndarray):
            self._values = torch.as_tensor(values, dtype=torch.double)
        elif isinstance(values, Sequence):
            # pyre-fixme[6]: Expected `Union[typing.List[Tensor],
            #  typing.Tuple[Tensor, ...]]` for 1st param but got `Sequence[Tensor]`.
            self._values = torch.stack(values).to(dtype=torch.double)
        elif isinstance(values, Mapping):
            self._key_to_index = dict(zip(values.keys(), range(len(values))))
            self._index_to_key = list(values.keys())
            self._values = torch.stack(list(values.values())).to(dtype=torch.double)
        else:
            raise TypeError(f"Unsupported values type {type(values)}")

    def _to_key(self, k: int) -> SlateItem:
        return SlateItem(k)

    @property
    def items(self) -> SlateItems:
        if self.is_sequence:
            return SlateItems(len(self))
        else:
            return SlateItems(super().keys)


# SlateSlotFeatures = SlateSlotObjects[Tensor]
class SlateSlotFeatures(SlateSlotObjects[Tensor]):
    @property
    def features(self) -> Tensor:
        # pyre-fixme[16]: `SlateSlotFeatures` has no attribute `_values`.
        return torch.stack(self._values)


class Slate(SlateSlotObjects[SlateItem]):
    """
    Class represents a slate: map from slots to items/docs
    """

    def one_hots(self, items: SlateItems, device=None) -> Tensor:
        oh = torch.zeros((len(self), len(items)), dtype=torch.double, device=device)
        # pyre-fixme[16]: `Slate` has no attribute `_values`.
        for t, i in zip(oh, self._values):
            t[items.index_of(i)] = 1.0
        return oh

    @property
    def items(self) -> Sequence[SlateItem]:
        return super().values

    def slot_values(self, item_values: SlateItemValues) -> SlateSlotValues:
        """
        Map items in the slate to given values
        Args:
            item_values: Map from all items to some values

        Returns:
            List of values in the slate
        """
        if self._key_to_index is None:
            # pyre-fixme[16]: `Slate` has no attribute `_values`.
            return SlateSlotValues([item_values[i] for i in self._values])
        else:
            return SlateSlotValues({k: item_values[i] for k, i in self._key_to_index})

    def slot_features(self, item_features: SlateItemFeatures) -> SlateSlotFeatures:
        """
        Map items in the slate to given values
        Args:
            item_values: Map from all items to some values

        Returns:
            List of values in the slate
        """
        if self._key_to_index is None:
            return SlateSlotFeatures(
                # pyre-fixme[16]: `Slate` has no attribute `_values`.
                [item_features[i].detach().clone() for i in self._values]
            )
        else:
            return SlateSlotFeatures(
                {k: item_features[i].detach().clone() for k, i in self._key_to_index}
            )

    def __repr__(self):
        return f"{self.__class__.__name__}{{value[{self._values}]}}"


def make_slate(slots: SlateSlots, items: Sequence[SlateItem]) -> Slate:
    """
    Assign items to slots to make a slate
    """
    assert len(items) >= len(slots)
    if slots.is_sequence:
        return Slate(list(items[: len(slots)]))
    else:
        return Slate(dict(zip(slots, items[: len(slots)])))


class SlateSlotItemValues(SlateSlotObjects[SlateItemValues]):
    def __init__(
        self,
        values: Union[
            MutableMapping[SlateSlot, SlateItemValues], MutableSequence[SlateItemValues]
        ],
    ):
        super().__init__(values)
        # pyre-fixme[16]: `SlateSlotItemValues` has no attribute `_values`.
        self._item_size = len(self._values[0])
        for v in self._values[1:]:
            assert self._item_size == len(v)

    def values_tensor(self, device=None) -> Tensor:
        # pyre-fixme[16]: `SlateSlotItemValues` has no attribute `_values`.
        dist = [v.values for v in self._values]
        return torch.stack(dist).to(device=device)


class SlateSlotItemExpectations(SlateSlotItemValues):
    def expected_rewards(
        self, item_rewards: SlateItemValues, device=None
    ) -> SlateSlotValues:
        """
        Calculate expected relevances of each slot, given each item's
        relevances, under this distribution
        Args:
            item_rewards:
            device:

        Returns:
            Map of slots to their expected relevance
        """
        dist = self.values_tensor(device)
        rewards = item_rewards.values.to(device=device)
        rewards = torch.mm(dist, rewards.unsqueeze(0).t()).squeeze()
        if self.is_sequence:
            return SlateSlotValues(rewards)
        else:
            return SlateSlotValues(dict(zip(self.slots, rewards.tolist())))

    @property
    def expectations(self) -> Sequence[SlateItemValues]:
        return super().values


def make_slot_item_distributions(
    slots: SlateSlots, dists: Sequence[SlateItemValues]
) -> SlateSlotItemExpectations:
    assert len(dists) >= len(slots)
    if slots.is_sequence:
        return SlateSlotItemExpectations(list(dists[: len(slots)]))
    else:
        return SlateSlotItemExpectations(dict(zip(slots, dists[: len(slots)])))


def is_to_calculate_expectation(slate_size: int, item_size: int) -> bool:
    """
    Switch between calculating and sampling expectations, balanced by execution
    time and accuracy
    Return:
        True to calculate
        False to sample
    """
    return (
        slate_size < 4
        or (slate_size == 4 and item_size < 182)
        or (slate_size == 5 and item_size < 47)
        or (slate_size == 6 and item_size < 22)
        or (slate_size == 7 and item_size < 15)
    )


def _calculate_slot_expectation(
    d_out: Tensor,
    probs: Sequence[float],
    buffer: Iterable[Tuple[Set[int], float, float, float]],
) -> Iterable[Tuple[Set[int], float, float, float]]:
    """
    A helper function to calculate items' expectations for a slot
    """
    assert d_out.shape[0] == len(probs)
    next_buffer = []
    for b0, b1, b2, _ in buffer:
        # memory buffer for all ordered combinations so far, list of tuples of
        #   b0: all the items in this ordered combination
        #   b1: cumulative probability of b0
        #   b2: sum of the probabilities of b0
        #   b3: = b1 / (1.0 - b2) cached value for faster computation
        for i, i_prob in enumerate(probs):
            # only add i if it's not already in
            if i in b0:
                continue
            # nb* are next buffer values
            nb2 = b2 + i_prob
            # due to precision errors, sometimes nb2 becomes 1, in this
            # case, discard the combination
            if nb2 < 1.0:
                nb1 = b1 * i_prob / (1.0 - b2)
                next_buffer.append(({*b0, i}, nb1, nb2, nb1 / (1.0 - nb2)))
    for i, i_prob in enumerate(probs):
        p = 0.0
        for b0, _, _, b3 in next_buffer:
            if i in b0:
                continue
            p += b3
        d_out[i] = p * i_prob
    return next_buffer


class SlateItemProbabilities(SlateItemValues):
    """
    Probabilities of each item being selected into the slate
    """

    def __init__(
        self,
        values: Union[Mapping[SlateItem, float], Sequence[float], np.ndarray, Tensor],
        greedy: bool = False,
    ):
        super().__init__(values)
        self._greedy = greedy
        self._slot_item_expectations = None

    def _to_key(self, k: int) -> SlateItem:
        return SlateItem(k)

    def _reset(self):
        super()._reset()
        self._slot_item_expectations = None

    def slate_probability(self, slate: Slate) -> Probability:
        """
        Calculate probability of a slate under this distribution
        Args:
            slate:

        Returns:
            probability
        """
        if self._greedy:
            items = super().greedy(len(slate))
            assert isinstance(items, Sequence)
            for i1, i2 in zip(items, slate.items):
                if i1 != i2:
                    return 0.0
            return 1.0
        else:
            # pyre-fixme[16]: `SlateItemProbabilities` has no attribute `_values`.
            clamped = torch.clamp(self._values, 0.0)
            indices = [self.index_of(item) for _, item in slate]
            probs = clamped[indices]
            sums = clamped[indices]
            clamped[indices] = 0.0
            sums = sums.flip(0).cumsum(0).flip(0) + clamped.sum()
            return Probability((probs / sums).prod().item())

    def slot_item_expectations(self, slots: SlateSlots) -> SlateSlotItemExpectations:
        slate_size = len(slots)
        if (
            self._slot_item_expectations is not None
            and len(self._slot_item_expectations) >= slate_size
        ):
            return self._slot_item_expectations
        item_size = len(self)
        assert item_size >= slate_size
        if self._greedy:
            self._slot_item_expectations = make_slot_item_distributions(
                slots,
                # pyre-fixme[6]: Expected `Sequence[SlateItemValues]` for 2nd param
                #  but got `List[Values[typing.Any]]`.
                [
                    self.replace(torch.zeros(item_size, dtype=torch.double))
                    for _ in range(len(self))
                ],
            )
            sorted_items, _ = self.sort()
            for item, ds in zip(
                sorted_items, self._slot_item_expectations.expectations
            ):
                ds[item] = 1.0
        else:
            self._normalize()
            if is_to_calculate_expectation(len(slots), len(self)):
                self._calculate_expectations(slots)
            else:
                self._sample_expectations(slots, 20000)
        return self._slot_item_expectations

    def _sample_expectations(self, slots: SlateSlots, num_samples: int):
        slate_size = len(slots)
        item_size = len(self)
        dm = torch.zeros((slate_size, item_size), dtype=torch.double)
        ri = torch.arange(slate_size)
        # pyre-fixme[16]: `SlateItemProbabilities` has no attribute `_probabilities`.
        ws = self._probabilities.repeat((num_samples, 1))
        for _ in range(item_size):
            samples = torch.multinomial(ws, slate_size)
            for sample in samples:
                dm[ri, sample] += 1
        dm /= num_samples * item_size
        self._slot_item_expectations = make_slot_item_distributions(
            slots,
            # pyre-fixme[6]: Expected `Sequence[SlateItemValues]` for 2nd param but
            #  got `List[Values[typing.Any]]`.
            [self.replace(vs) for vs in dm],
        )

    def _calculate_expectations(self, slots: SlateSlots):
        """
        A brute-force way to calculate each item's expectations at each slot by
        going through all l-choose-m (l!/(l-m)!) possible slates.
        """
        slate_size = len(slots)
        item_size = len(self)
        dm = torch.zeros((slate_size, item_size), dtype=torch.double)
        # pyre-fixme[16]: `SlateItemProbabilities` has no attribute `_probabilities`.
        dm[0] = self._probabilities
        buffer = [(set(), 1.0, 0.0, 1.0)]
        probs = self._probabilities.tolist()
        for d in dm[1:]:
            buffer = _calculate_slot_expectation(d, probs, buffer)
        self._slot_item_expectations = make_slot_item_distributions(
            slots,
            # pyre-fixme[6]: Expected `Sequence[SlateItemValues]` for 2nd param but
            #  got `List[Values[typing.Any]]`.
            [self.replace(vs) for vs in dm],
        )

    def sample_slate(self, slots: SlateSlots) -> Slate:
        slate_size = len(slots)
        if self._greedy:
            items = super().greedy(slate_size)
        else:
            items = super().sample(slate_size)
        if slate_size == 1:
            items = [items]
        # pyre-fixme[6]: Expected `Sequence[TypeWrapper[Union[Tuple[float],
        #  Tuple[int], Tensor, float, int, np.ndarray]]]` for 2nd param but got
        #  `Union[Sequence[Union[Sequence[TypeWrapper[Union[Tuple[float], Tuple[int],
        #  Tensor, float, int, np.ndarray]]], TypeWrapper[Union[Tuple[float],
        #  Tuple[int], Tensor, float, int, np.ndarray]]]],
        #  TypeWrapper[Union[Tuple[float], Tuple[int], Tensor, float, int,
        #  np.ndarray]]]`.
        return make_slate(slots, items)

    @property
    def is_deterministic(self) -> bool:
        return self._greedy

    def slate_space(
        self, slots: SlateSlots, max_size: int = -1
    ) -> Iterable[Tuple[Sequence[SlateItem], float]]:
        """Return all possible slates and their probabilities

        The algorithm is similar to :func:`~_calculate_expectations`, but has
        less value to cache thus save both space and computation
        Args:
            slots: slots to be filled
            max_size: max number of samples to be returned
                      <= 0 return all samples
        """
        slate_size = len(slots)
        item_size = len(self)
        assert item_size >= slate_size
        if self._greedy:
            items = super().greedy(slate_size)
            assert isinstance(items, Sequence)
            return [(items, 1.0)]
        else:
            buffer = [([], 1.0, 0.0)]
            # pyre-fixme[16]: `SlateItemProbabilities` has no attribute
            #  `_probabilities`.
            probs = self._probabilities.tolist()
            for _ in range(slate_size):
                next_buffer = []
                for b0, b1, b2 in buffer:
                    # memory buffer for all ordered combinations so far, list of tuples of
                    #   b0: all the items in this ordered combination
                    #   b1: cumulative probability of b0
                    #   b2: sum of the probabilities of b0
                    for i, i_prob in enumerate(probs):
                        if i in b0:
                            continue
                        nb2 = b2 + i_prob
                        if nb2 < 1.0:
                            nb1 = b1 * i_prob / (1.0 - b2)
                            next_buffer.append(([*b0, i], nb1, nb2))
                if max_size <= 0 or max_size > len(next_buffer):
                    buffer = next_buffer
                else:
                    buffer = random.sample(next_buffer, max_size)
            return [([SlateItem(i) for i in b[0]], b[1]) for b in buffer]


class SlateSlotItemProbabilities(SlateSlotItemValues):
    def __init__(
        self,
        values: Union[
            MutableMapping[SlateSlot, SlateItemValues], MutableSequence[SlateItemValues]
        ],
        greedy: bool = False,
    ):
        super().__init__(values)
        self._greedy = greedy
        self._slot_item_distributions = None
        self._slot_item_expectations = None

    def slate_probability(self, slate: Slate) -> Probability:
        """
        Calculate probability of a slate under this distribution
        Args:
            slate:

        Returns:
            probability
        """
        assert len(slate) <= len(self)
        if self._greedy:
            for slot, item in slate:
                probs = self[slot]
                its, _ = probs.sort()
                if its[0] != item:
                    return 0.0
            return 1.0
        else:
            p = 1.0
            last_items = []
            for slot, item in slate:
                item_probs = self[slot]
                w = 1.0
                for last_item in last_items:
                    w -= item_probs.probability(last_item)
                if math.fabs(w - 0.0) < 1.0e-10:
                    return 0.0
                p *= item_probs.probability(item) / w
                last_items.append(item)
            return p

    def slot_item_expectations(self, samples: int = 20000) -> SlateSlotItemExpectations:
        slate_size = len(self.slots)
        if (
            self._slot_item_expectations is not None
            and len(self._slot_item_expectations) >= slate_size
        ):
            return self._slot_item_expectations
        # pyre-fixme[16]: `SlateSlotItemProbabilities` has no attribute `_values`.
        item_size = len(self._values[0])
        assert item_size >= slate_size
        ps = self.values_tensor()
        if self._greedy:
            dists = []
            for i, value in zip(range(slate_size), self._values):
                item = ps[i].argmax().item()
                dist = torch.zeros(item_size, dtype=torch.double)
                dist[item] = 1.0
                dists.append(value.replace(dist))
                ps[torch.arange(i + 1, slate_size), item] = 0.0
            self._slot_item_expectations = make_slot_item_distributions(
                self.slots, dists
            )
        else:
            if is_to_calculate_expectation(slate_size, item_size):
                self._calculate_expectations()
            else:
                self._sample_expectations(samples * item_size)
        return self._slot_item_expectations

    def _sample_expectations(self, num_samples: int):
        slate_size = len(self.slots)
        # pyre-fixme[16]: `SlateSlotItemProbabilities` has no attribute `_values`.
        item_size = len(self._values[0])
        dm = torch.zeros((slate_size, item_size), dtype=torch.double)
        ri = torch.arange(slate_size)
        for _ in range(num_samples):
            ps = self.values_tensor()
            sample = []
            for i in range(slate_size):
                item = ps[i].multinomial(1)
                sample.append(item)
                ps[torch.arange(i + 1, slate_size), item] = 0.0
            dm[ri, sample] += 1
        dm /= num_samples
        self._slot_item_expectations = make_slot_item_distributions(
            self.slots, [ivs.replace(vs) for ivs, vs in zip(self._values, dm)]
        )

    def _calculate_expectations(self):
        slate_size = len(self.slots)
        item_size = len(self._values[0])
        dm = torch.zeros((slate_size, item_size), dtype=torch.double)
        prob_list = []
        for v in self._values:
            v._normalize()
            prob_list.append(v._probabilities.detach().clone())
        dm[0] = prob_list[0]
        buffer = [({}, 1.0, 0.0, 1.0)]
        for d, probs in zip(dm[1:], prob_list[1:]):
            buffer = _calculate_slot_expectation(d, probs.tolist(), buffer)
        self._slot_item_expectations = make_slot_item_distributions(
            self.slots, [its.replace(vs) for its, vs in zip(self._values, dm)]
        )

    def sample_slate(self, slots: SlateSlots) -> Slate:
        slate_size = len(slots)
        ps = self.values_tensor()
        items = []
        if self._greedy:
            # pyre-fixme[16]: `SlateSlotItemProbabilities` has no attribute `_values`.
            for i, value in zip(range(slate_size), self._values):
                item = ps[i].argmax().item()
                items.append(value.items[item])
                ps[torch.arange(i + 1, slate_size), item] = 0.0
        else:
            for i, value in zip(range(slate_size), self._values):
                item = ps[i].multinomial(1).item()
                items.append(value.items[item])
                ps[torch.arange(i + 1, slate_size), item] = 0.0
        return make_slate(slots, items)


class RewardDistribution(ABC):
    """
    Return customized probability distribution according to rewards
    """

    def __init__(self, deterministic: bool = False):
        self._deterministic = deterministic

    @abstractmethod
    def distribution(self, rewards: Tensor) -> Tensor:
        pass

    def __call__(self, rewards: SlateItemValues) -> SlateItemProbabilities:
        dist = self.distribution(rewards.values)
        return SlateItemProbabilities(rewards.items.fill(dist), self._deterministic)

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class PassThruDistribution(RewardDistribution):
    """
    No-op distribution, probability determined by reward
    """

    def distribution(self, rewards: Tensor) -> Tensor:
        return rewards.detach().clone()

    @property
    def name(self) -> str:
        return f"{self._deterministic}"

    def __repr__(self):
        return f"PassThruDistribution[deterministic={self._deterministic}]"


class RankingDistribution(RewardDistribution):
    """
    Ranking distribution according to https://arxiv.org/abs/1605.04812
    """

    def __init__(self, alpha: float = -1.0, deterministic: bool = False):
        super().__init__(deterministic)
        self._alpha = alpha

    def distribution(self, rewards: Tensor) -> Tensor:
        dist = rewards.detach().clone()
        if self._alpha >= 0:
            _, ids = torch.sort(rewards, descending=True)
            rank = torch.arange(1, ids.shape[0] + 1, dtype=torch.double)
            dist[ids] = torch.pow(
                2.0,
                # pyre-fixme[16]: `float` has no attribute `floor_`.
                (-1.0 * (self._alpha * torch.log2(rank)).floor_()),
            )
        return dist

    @property
    def name(self) -> str:
        return f"ranking_{self._alpha}_{self._deterministic}"

    def __repr__(self):
        return (
            f"RankingDistribution[alpha={self._alpha}"
            f",deterministic={self._deterministic}]"
        )


class FrechetDistribution(RewardDistribution):
    """
    Frechet distribution
    """

    def __init__(self, shape: float, deterministic: bool = False):
        super().__init__(deterministic)
        self._shape = shape

    def distribution(self, rewards: Tensor) -> Tensor:
        return torch.pow(rewards, self._shape)

    @property
    def name(self) -> str:
        return f"frechet_{self._shape}_{self._deterministic}"

    def __repr__(self):
        return (
            f"FrechetDistribution[shape={self._shape}]"
            f",deterministic={self._deterministic}]"
        )


SlateQueryType = Union[Tuple[int], Tuple[float], np.ndarray, Tensor, Tuple[int, int]]
SlateQuery = TypeWrapper[SlateQueryType]


@dataclass(frozen=True)
class SlateContext:
    query: SlateQuery
    slots: SlateSlots
    params: object = None


class SlatePolicy(ABC):
    """
    Policy interface
    """

    def __init__(self, device=None):
        self.device = device

    @abstractmethod
    def _query(self, context: SlateContext) -> SlateItemProbabilities:
        pass

    def __call__(self, context: SlateContext) -> SlateItemProbabilities:
        return self._query(context)


class SlateMetric:
    """
    Metric calculator for a slate: weights (dot) rewards

    Base class is just sum of the all item rewards
    """

    def __init__(self, device=None):
        self._device = device

    def calculate_reward(
        self,
        slots: SlateSlots,
        rewards: Optional[SlateSlotValues] = None,
        slot_values: Optional[SlateSlotValues] = None,
        slot_weights: Optional[SlateSlotValues] = None,
    ) -> float:
        if slot_values is None:
            assert rewards is not None
            slot_values = self.slot_values(rewards)
        values = slot_values.values.to(device=self._device)
        if slot_weights is None:
            slot_weights = self.slot_weights(slots)
        weights = slot_weights.values.to(device=self._device)
        return torch.tensordot(values, weights, dims=([0], [0])).item()

    def __call__(self, slots: SlateSlots, rewards: SlateSlotValues) -> float:
        return self.calculate_reward(slots, rewards)

    def slot_weights(self, slots: SlateSlots) -> SlateSlotValues:
        return slots.fill([1.0] * len(slots))

    def slot_values(self, rewards: SlateSlotValues) -> SlateSlotValues:
        return rewards


class DCGSlateMetric(SlateMetric):
    _weights: Optional[Tensor] = None

    def _get_discount(self, slate_size: int) -> Tensor:
        weights = DCGSlateMetric._weights
        if (
            weights is None
            or weights.shape[0] < slate_size
            or weights.device != self._device
        ):
            DCGSlateMetric._weights = torch.reciprocal(
                torch.log2(
                    torch.arange(
                        2, slate_size + 2, dtype=torch.double, device=self._device
                    )
                )
            )
        weights = DCGSlateMetric._weights
        assert weights is not None
        return weights[:slate_size]

    def slot_weights(self, slots: SlateSlots) -> SlateSlotValues:
        return slots.fill(self._get_discount(len(slots)))

    def slot_values(self, rewards: SlateSlotValues) -> SlateSlotValues:
        # pyre-fixme[7]: Expected `SlateSlotValues` but got `Values[typing.Any]`.
        return rewards.replace(torch.pow(2.0, rewards.values) - 1.0)


class NDCGSlateMetric(DCGSlateMetric):
    def __init__(self, item_rewards: SlateItemValues, device=None):
        super().__init__(device)
        self._sorted_items, _ = item_rewards.sort()
        self._item_rewards = item_rewards
        self._idcg = {}

    def slot_weights(self, slots: SlateSlots) -> SlateSlotValues:
        slate_size = len(slots)
        assert len(self._sorted_items) >= slate_size
        if slate_size not in self._idcg:
            i_slate = make_slate(slots, self._sorted_items[:slate_size])
            idcg = super().calculate_reward(
                slots,
                i_slate.slot_values(self._item_rewards),
                None,
                super().slot_weights(slots),
            )
            self._idcg[slate_size] = idcg
        else:
            idcg = self._idcg[slate_size]
        return slots.fill(
            torch.zeros(slate_size, dtype=torch.double)
            if idcg == 0
            else self._get_discount(slate_size) / idcg
        )


class ERRSlateMetric(SlateMetric):
    def __init__(self, max_reward: float, device=None):
        super().__init__(device)
        self._max_reward = max_reward

    def slot_weights(self, slots: SlateSlots) -> SlateSlotValues:
        return slots.fill([1.0 / (r + 1) for r in range(len(slots))])

    def slot_values(self, rewards: SlateSlotValues) -> SlateSlotValues:
        d = torch.tensor(self._max_reward, device=self._device).pow(2.0)
        r = (torch.pow(2.0, rewards.values.clamp(0.0, self._max_reward)) - 1.0) / d
        p = 1.0
        err = torch.zeros(len(rewards), dtype=torch.double, device=self._device)
        for i in range(len(rewards)):
            ri = r[i]
            err[i] = p * ri
            p = p * (1.0 - ri.item())
        # pyre-fixme[7]: Expected `SlateSlotValues` but got `Values[typing.Any]`.
        return rewards.replace(err)


class SlateModel(ABC):
    """
    Model providing item relevance/reward, slot examination (click) distribution
    """

    @abstractmethod
    def item_rewards(self, context: SlateContext) -> SlateItemValues:
        """
        Returns each item's relevance under the context
        Args:
            context:

        Returns:
            Item relevances
        """
        pass

    def slot_probabilities(self, context: SlateContext) -> SlateSlotValues:
        """
        Returns each slot/positions's probability independent of showing item,
        used in PBM estimator
        Args:
            context:

        Returns:

        """
        return context.slots.fill(torch.ones(len(context.slots), dtype=torch.double))


@dataclass(frozen=True)
class LogSample:
    context: SlateContext
    metric: SlateMetric
    log_slate: Slate
    log_reward: Reward
    _log_slate_probability: Probability = float("nan")
    # probability for each item being places at each slot
    _log_slot_item_probabilities: Optional[SlateSlotItemProbabilities] = None
    # item probability distribution from behavior policy
    _log_item_probabilities: Optional[SlateItemProbabilities] = None
    _tgt_slate_probability: Probability = float("nan")
    _tgt_slot_item_probabilities: Optional[SlateSlotItemProbabilities] = None
    # item probability distribution from target policy
    _tgt_item_probabilities: Optional[SlateItemProbabilities] = None
    # gt_item_rewards: Optional[SlateItemValues] = None
    # pre-calculated ground truth for target policy
    ground_truth_reward: Reward = float("nan")
    # context dependent slot weights (e.g. DCG or ERR weights), used by PBM
    slot_weights: Optional[SlateSlotValues] = None
    # item/action independent examination probabilities of each slot, used by PBM
    slot_probabilities: Optional[SlateSlotValues] = None
    # features associated with the slate, to train direct model
    item_features: Optional[SlateItemFeatures] = None

    def validate(self):
        slate_size = len(self.context.slots)
        item_size = len(self.items)
        assert len(self.log_slate) == slate_size
        assert (
            math.isnan(self._log_slate_probability)
            or self._log_slate_probability <= 1.0
        )
        assert (
            math.isnan(self._tgt_slate_probability)
            or self._tgt_slate_probability <= 1.0
        )
        assert (
            self._log_slot_item_probabilities is None
            or len(self._log_slot_item_probabilities) == slate_size
        )
        assert (
            self._log_item_probabilities is None
            or len(self._log_item_probabilities) == item_size
        )
        assert (
            self._tgt_slot_item_probabilities is None
            or len(self._tgt_slot_item_probabilities) == slate_size
        )
        assert (
            self._tgt_item_probabilities is None
            or len(self._tgt_item_probabilities) == item_size
        )
        assert self.slot_weights is None or len(self.slot_weights) == slate_size
        assert (
            self.slot_probabilities is None
            or len(self.slot_probabilities) == slate_size
        )

    def log_slot_item_expectations(
        self, slots: SlateSlots
    ) -> Optional[SlateSlotItemExpectations]:
        if self._log_slot_item_probabilities is not None:
            return self._log_slot_item_probabilities.slot_item_expectations()
        if self._log_item_probabilities is not None:
            return self._log_item_probabilities.slot_item_expectations(slots)
        return None

    def log_slate_probability(self, slate: Optional[Slate] = None) -> float:
        if not math.isnan(self._log_slate_probability):
            return self._log_slate_probability
        if slate is None:
            slate = self.log_slate
        if self._log_slot_item_probabilities is not None:
            return self._log_slot_item_probabilities.slate_probability(slate)
        if self._log_item_probabilities is not None:
            return self._log_item_probabilities.slate_probability(slate)
        return 0.0

    def tgt_slot_expectations(
        self, slots: SlateSlots
    ) -> Optional[SlateSlotItemExpectations]:
        if self._tgt_slot_item_probabilities is not None:
            return self._tgt_slot_item_probabilities.slot_item_expectations()
        if self._tgt_item_probabilities is not None:
            return self._tgt_item_probabilities.slot_item_expectations(slots)
        return None

    def tgt_slate_probability(self) -> float:
        if not math.isnan(self._tgt_slate_probability):
            return self._tgt_slate_probability
        if self._tgt_slot_item_probabilities is not None:
            return self._tgt_slot_item_probabilities.slate_probability(self.log_slate)
        if self._tgt_item_probabilities is not None:
            return self._tgt_item_probabilities.slate_probability(self.log_slate)
        return 0.0

    def tgt_slate_space(
        self, slots: SlateSlots
    ) -> Iterable[Tuple[Sequence[SlateItem], float]]:
        if self._tgt_item_probabilities is not None:
            return self._tgt_item_probabilities.slate_space(slots)
        return []

    @property
    def items(self) -> SlateItems:
        if self._log_slot_item_probabilities is not None:
            # pyre-fixme[16]: `SlateSlotItemProbabilities` has no attribute `_values`.
            return self._log_slot_item_probabilities._values[0].items
        if self._log_item_probabilities is not None:
            return self._log_item_probabilities.items
        return SlateItems(0)


@dataclass(frozen=True)
class SlateEstimatorInput:
    samples: Sequence[LogSample]

    def validate(self):
        for s in self.samples:
            s.validate()


class SlateEstimator(Estimator):
    @abstractmethod
    def _evaluate_sample(self, sample: LogSample) -> Optional[EstimatorSampleResult]:
        pass


class DMEstimator(SlateEstimator):
    """
    Direct Method estimator
    """

    def __init__(self, trainer: Trainer, training_sample_ratio: float, device=None):
        super().__init__(device)
        self._trainer = trainer
        self._training_sample_ratio = training_sample_ratio

    def _train_model(
        self, samples: Sequence[LogSample]
    ) -> Optional[Iterable[LogSample]]:
        if self._trainer is None:
            logger.error("Target model trainer is none, DM is not available")
            return None
        self._trainer.reset()
        logger.info("  training direct model...")
        st = time.perf_counter()
        sample_size = len(samples)
        if self._training_sample_ratio > 0.0 and self._training_sample_ratio < 1.0:
            training_samples = range(int(sample_size * self._training_sample_ratio))
        else:
            training_samples = range(sample_size)
        train_x = []
        train_y = []
        vali_mask = [True] * len(samples)
        for i in training_samples:
            sample = samples[i]
            if sample.item_features is None:
                continue
            slate_features = sample.log_slate.slot_features(sample.item_features)
            train_x.append(slate_features.features.flatten())
            train_y.append(sample.log_reward)
            vali_mask[i] = False
        if len(train_x) == 0:
            logger.error("Slate features not provided, DM is not available")
            return None
        train_x = torch.stack(train_x)
        train_y = torch.tensor(train_y, dtype=torch.double, device=train_x.device)
        vali_x = []
        vali_y = []
        evaluate_samples = []
        for mask, sample in zip(vali_mask, samples):
            if not mask or sample.item_features is None:
                continue
            slate_features = sample.log_slate.slot_features(sample.item_features)
            vali_x.append(slate_features.features.flatten())
            vali_y.append(sample.log_reward)
            evaluate_samples.append(sample)
        if len(vali_x) == 0:
            vali_x = train_x.detach().clone()
            vali_y = train_y.detach().clone()
            evaluate_samples = samples
        else:
            vali_x = torch.stack(vali_x)
            vali_y = torch.tensor(vali_y, dtype=torch.double, device=vali_x.device)
        training_data = TrainingData(train_x, train_y, None, vali_x, vali_y, None)
        self._trainer.train(training_data)
        logger.info(f"  training direct model done: {time.perf_counter() - st}s")

        return evaluate_samples

    def _evaluate_sample(self, sample: LogSample) -> Optional[EstimatorSampleResult]:
        slots = sample.context.slots
        tgt_slate_space = sample.tgt_slate_space(slots)
        features = []
        probs = []
        for items, prob in tgt_slate_space:
            slate = make_slate(slots, items)
            assert sample.item_features is not None
            slate_features = slate.slot_features(sample.item_features)
            features.append(slate_features.features.flatten())
            probs.append(prob)
        preds = self._trainer.predict(torch.stack(features), device=self._device)
        tgt_reward = torch.dot(
            preds.scores, torch.tensor(probs, dtype=torch.double, device=self._device)
        )
        return EstimatorSampleResult(
            sample.log_reward,
            tgt_reward.item(),
            sample.ground_truth_reward,
            float("nan"),
        )

    # pyre-fixme[14]: `evaluate` overrides method defined in `Estimator` inconsistently.
    def evaluate(
        self, input: SlateEstimatorInput, *kwargs
    ) -> Optional[EstimatorResult]:
        input.validate()
        samples = self._train_model(input.samples)
        if samples is None:
            return None

        log_avg = RunningAverage()
        tgt_avg = RunningAverage()
        gt_avg = RunningAverage()
        for sample in samples:
            result = self._evaluate_sample(sample)
            if result is None:
                continue
            log_avg.add(result.log_reward)
            tgt_avg.add(result.target_reward)
            gt_avg.add(result.ground_truth_reward)
        return EstimatorResult(
            log_avg.average, tgt_avg.average, gt_avg.average, tgt_avg.count
        )

    def __repr__(self):
        return (
            f"DMEstimator(trainer({self._trainer.name})"
            f",ratio({self._training_sample_ratio}),device({self._device}))"
        )


class IPSEstimator(SlateEstimator):
    def __init__(
        self,
        weight_clamper: Optional[Clamper] = None,
        weighted: bool = True,
        device=None,
    ):
        super().__init__(device)
        self._weight_clamper = (
            weight_clamper if weight_clamper is not None else Clamper()
        )
        self._weighted = weighted

    def _evaluate_sample(self, sample: LogSample) -> Optional[EstimatorSampleResult]:
        tgt_prob = sample.tgt_slate_probability()
        log_prob = sample.log_slate_probability(sample.log_slate)
        if tgt_prob == log_prob:
            weight = 1.0
        elif tgt_prob <= 0.0:
            weight = 0.0
        elif log_prob <= 0.0:
            return None
        else:
            weight = self._weight_clamper(tgt_prob / log_prob)
        return EstimatorSampleResult(
            sample.log_reward,
            sample.log_reward * weight,
            sample.ground_truth_reward,
            weight,
        )

    # pyre-fixme[14]: `evaluate` overrides method defined in `Estimator` inconsistently.
    def evaluate(
        self, input: SlateEstimatorInput, *kwargs
    ) -> Optional[EstimatorResult]:
        input.validate()
        log_avg = RunningAverage()
        tgt_avg = RunningAverage()
        acc_weight = RunningAverage()
        gt_avg = RunningAverage()
        zw = 0
        for sample in input.samples:
            result = self._evaluate_sample(sample)
            if result is None:
                zw += 1
                continue
            log_avg.add(result.log_reward)
            tgt_avg.add(result.target_reward)
            gt_avg.add(result.ground_truth_reward)
            acc_weight.add(result.weight)
            if result.weight == 0.0:
                zw += 1
        logging.info(
            f"IPSEstimator invalid sample pct: {zw * 100 / len(input.samples)}%"
        )
        if tgt_avg.count == 0:
            return None
        if self._weighted:
            estimated = tgt_avg.total / acc_weight.total
            return EstimatorResult(
                log_avg.average, estimated, gt_avg.average, acc_weight.average
            )
        else:
            return EstimatorResult(
                log_avg.average, tgt_avg.average, gt_avg.average, tgt_avg.count
            )

    def __repr__(self):
        return (
            f"IPSEstimator(weight_clamper({self._weight_clamper})"
            f",weighted({self._weighted}),device({self._device}))"
        )


class DoublyRobustEstimator(DMEstimator):
    def __init__(
        self,
        trainer: Trainer,
        training_sample_ratio: float,
        weight_clamper: Optional[Clamper] = None,
        weighted: bool = False,
        device=None,
    ):
        super().__init__(trainer, training_sample_ratio, device)
        self._weight_clamper = (
            weight_clamper if weight_clamper is not None else Clamper()
        )
        self._weighted = weighted

    def _evaluate_sample(self, sample: LogSample) -> Optional[EstimatorSampleResult]:
        slots = sample.context.slots
        if self._trainer.is_trained:
            tgt_slate_space = sample.tgt_slate_space(slots)
            features = []
            probs = []
            for items, prob in tgt_slate_space:
                slate = make_slate(slots, items)
                assert sample.item_features is not None
                slate_features = slate.slot_features(sample.item_features)
                features.append(slate_features.features.flatten())
                probs.append(prob)
            preds = self._trainer.predict(torch.stack(features), device=self._device)
            dm_reward = torch.dot(
                preds.scores,
                torch.tensor(probs, dtype=torch.double, device=self._device),
            ).item()
            assert sample.item_features is not None
            log_slate_feature = sample.log_slate.slot_features(sample.item_features)
            pred = self._trainer.predict(
                torch.unsqueeze(log_slate_feature.features.flatten(), dim=0),
                device=self._device,
            )
            log_dm_reward = pred.scores[0].item()
        else:
            dm_reward = 0.0
            log_dm_reward = 0.0
        tgt_prob = sample.tgt_slate_probability()
        log_prob = sample.log_slate_probability(sample.log_slate)
        if tgt_prob == log_prob:
            weight = 1.0
        elif tgt_prob <= 0.0:
            weight = 0.0
        elif log_prob <= 0.0:
            return None
        else:
            weight = self._weight_clamper(tgt_prob / log_prob)
        target_reward = (sample.log_reward - log_dm_reward) * weight + dm_reward
        return EstimatorSampleResult(
            sample.log_reward, target_reward, sample.ground_truth_reward, weight
        )

    def evaluate(
        self, input: SlateEstimatorInput, *kwargs
    ) -> Optional[EstimatorResult]:
        input.validate()
        samples = self._train_model(input.samples)
        if samples is None:
            samples = input.samples

        log_avg = RunningAverage()
        tgt_avg = RunningAverage()
        acc_weight = RunningAverage()
        gt_avg = RunningAverage()
        for sample in samples:
            result = self._evaluate_sample(sample)
            if result is None:
                continue
            log_avg.add(result.log_reward)
            tgt_avg.add(result.target_reward)
            acc_weight.add(result.weight)
            gt_avg.add(result.ground_truth_reward)
        if self._weighted:
            estimated = tgt_avg.total / acc_weight.total
            return EstimatorResult(
                log_avg.average, estimated, gt_avg.average, acc_weight.average
            )
        else:
            return EstimatorResult(
                log_avg.average, tgt_avg.average, gt_avg.average, tgt_avg.count
            )

    def __repr__(self):
        return (
            f"DoublyRobustEstimator(trainer({self._trainer.name})"
            f",ratio({self._training_sample_ratio})"
            f",weight_clamper({self._weight_clamper})"
            f",weighted({self._weighted}),device({self._device}))"
        )


class PseudoInverseEstimator(SlateEstimator):
    """
    Estimator from reference 2
    """

    def __init__(
        self,
        weight_clamper: Optional[Clamper] = None,
        weighted: bool = True,
        device=None,
    ):
        super().__init__(device)
        self._weight_clamper = (
            weight_clamper if weight_clamper is not None else Clamper()
        )
        self._weighted = weighted

    def _evaluate_sample(self, sample: LogSample) -> Optional[EstimatorSampleResult]:
        log_slot_expects = sample.log_slot_item_expectations(sample.context.slots)
        if log_slot_expects is None:
            logger.warning("Log slot distribution not available")
            return None
        tgt_slot_expects = sample.tgt_slot_expectations(sample.context.slots)
        if tgt_slot_expects is None:
            logger.warning("Target slot distribution not available")
            return None
        log_indicator = log_slot_expects.values_tensor(self._device)
        tgt_indicator = tgt_slot_expects.values_tensor(self._device)
        lm = len(sample.context.slots) * len(sample.items)
        gamma = torch.as_tensor(
            np.linalg.pinv(
                torch.mm(
                    log_indicator.view((lm, 1)), log_indicator.view((1, lm))
                ).numpy()
            )
        )
        # torch.pinverse is not very stable
        # gamma = torch.pinverse(
        #     torch.mm(log_indicator.view((lm, 1)), log_indicator.view((1, lm)))
        # )
        ones = sample.log_slate.one_hots(sample.items, self._device)
        weight = self._weight_clamper(
            torch.mm(tgt_indicator.view((1, lm)), torch.mm(gamma, ones.view((lm, 1))))
        ).item()
        return EstimatorSampleResult(
            sample.log_reward,
            sample.log_reward * weight,
            sample.ground_truth_reward,
            weight,
        )

    # pyre-fixme[14]: `evaluate` overrides method defined in `Estimator` inconsistently.
    def evaluate(
        self, input: SlateEstimatorInput, *kwargs
    ) -> Optional[EstimatorResult]:
        input.validate()
        log_avg = RunningAverage()
        tgt_avg = RunningAverage()
        acc_weight = RunningAverage()
        gt_avg = RunningAverage()
        zw = 0
        for sample in input.samples:
            result = self._evaluate_sample(sample)
            if result is None:
                zw += 1
                continue
            log_avg.add(result.log_reward)
            tgt_avg.add(result.target_reward)
            gt_avg.add(result.ground_truth_reward)
            acc_weight.add(result.weight)
            if result.weight == 0.0:
                zw += 1
            if tgt_avg.count % 1000 == 0:
                logger.info(f"  PseudoInverseEstimator: processed {tgt_avg.count}")
        logging.info(
            f"PseudoInverseEstimator invalid sample pct: {zw * 100 / len(input.samples)}%"
        )
        if tgt_avg.count == 0:
            return None
        if self._weighted:
            estimated = tgt_avg.total / acc_weight.total
            return EstimatorResult(
                log_avg.average, estimated, gt_avg.average, acc_weight.average
            )
        else:
            return EstimatorResult(
                log_avg.average, tgt_avg.average, gt_avg.average, tgt_avg.count
            )

    def __repr__(self):
        return (
            f"PseudoInverseEstimator(weight_clamper({self._weight_clamper})"
            f",weighted({self._weighted}),device({self._device}))"
        )


class PBMEstimator(SlateEstimator):
    """
    Estimator from reference 1: Position-Based Click Model
    """

    def __init__(
        self,
        weight_clamper: Optional[Clamper] = None,
        weighted: bool = True,
        device=None,
    ):
        super().__init__(device)
        self._weight_clamper = (
            weight_clamper if weight_clamper is not None else Clamper()
        )
        self._weighted = weighted

    def _evaluate_sample(self, sample: LogSample) -> Optional[EstimatorSampleResult]:
        log_slot_expects = sample.log_slot_item_expectations(sample.context.slots)
        if log_slot_expects is None:
            logger.warning("  Log slot distribution not available")
            return None
        tgt_slot_expects = sample.tgt_slot_expectations(sample.context.slots)
        if tgt_slot_expects is None:
            logger.warning("  Target slot distribution not available")
            return None
        slate_size = len(sample.context.slots)
        slot_weights = sample.slot_weights
        if slot_weights is None:
            slot_weights = SlateSlotValues(torch.ones(slate_size, dtype=torch.double))
        weights = slot_weights.values.to(device=self._device)
        if sample.slot_probabilities is not None:
            weights *= sample.slot_probabilities.values
        h = torch.zeros(slate_size, dtype=torch.double, device=self._device)
        p = torch.zeros(slate_size, dtype=torch.double, device=self._device)
        i = 0
        for slot, item in sample.log_slate:
            h[i] = tgt_slot_expects[slot][item]
            p[i] = log_slot_expects[slot][item]
            i += 1
        nu = torch.tensordot(h, weights, dims=([0], [0]))
        de = torch.tensordot(p, weights, dims=([0], [0]))
        if nu == de:
            weight = 1.0
        elif nu == 0:
            weight = 0.0
        elif de == 0:
            return None
        else:
            weight = self._weight_clamper(nu / de)
        return EstimatorSampleResult(
            sample.log_reward,
            sample.log_reward * weight,
            sample.ground_truth_reward,
            weight,
        )

    # pyre-fixme[14]: `evaluate` overrides method defined in `Estimator` inconsistently.
    def evaluate(
        self, input: SlateEstimatorInput, *kwargs
    ) -> Optional[EstimatorResult]:
        input.validate()
        log_avg = RunningAverage()
        tgt_avg = RunningAverage()
        acc_weight = RunningAverage()
        gt_avg = RunningAverage()
        zw = 0
        for sample in input.samples:
            result = self._evaluate_sample(sample)
            if result is None:
                zw += 1
                continue
            log_avg.add(result.log_reward)
            tgt_avg.add(result.target_reward)
            gt_avg.add(result.ground_truth_reward)
            acc_weight.add(result.weight)
            if result.weight == 0.0:
                zw += 1
            if tgt_avg.count % 1000 == 0:
                logger.info(f"  PBMEstimator: processed {tgt_avg.count}")
        logging.info(
            f"PBMEstimator invalid sample pct: {zw * 100 / len(input.samples)}%"
        )
        if tgt_avg.count == 0:
            return None
        if self._weighted:
            estimated = tgt_avg.total / acc_weight.total
            return EstimatorResult(
                log_avg.average, estimated, gt_avg.average, acc_weight.average
            )
        else:
            return EstimatorResult(
                log_avg.average, tgt_avg.average, gt_avg.average, tgt_avg.count
            )

    def __repr__(self):
        return (
            f"PBMEstimator(weight_clamper({self._weight_clamper})"
            f",weighted({self._weighted}),device({self._device}))"
        )

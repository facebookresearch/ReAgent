#!/usr/bin/env python3

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Generic,
    Iterable,
    Mapping,
    MutableMapping,
    MutableSequence,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import torch
from reagent.ope.estimators.estimator import Estimator, EstimatorResults
from reagent.ope.estimators.types import (
    Action,
    Items,
    Probability,
    Type,
    TypeWrapper,
    Values,
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


class SlateSlotValues(Values[SlateSlot]):
    """
    Map from a slot to a value
    """

    def _new_key(self, k: int) -> SlateSlot:
        return SlateSlot(k)


class SlateSlots(Items[SlateSlot]):
    """
    List of slot
    """

    def _new_item(self, i: int) -> SlateSlot:
        return SlateSlot(i)

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
        return SlateSlotValues(super()._fill(values))


class SlateSlotObjects(Generic[Type]):
    def __init__(
        self, values: Union[MutableMapping[SlateSlot, Type], MutableSequence[Type]]
    ):
        assert (len(values)) > 0
        self._slot_to_index = None
        if isinstance(values, Mapping):
            self._slot_to_index = {s: i for i, s in enumerate(values.keys())}
            self._values = list(values.values())
        else:
            self._values = values

    def __getitem__(self, slot: SlateSlot) -> Optional[Type]:
        try:
            if self._slot_to_index is None:
                return self._values[slot]
            else:
                return self._values[self._slot_to_index[slot]]
        except Exception:
            return None

    def __setitem__(self, slot: SlateSlot, value: Type):
        if self._slot_to_index is None:
            self._values[slot] = value
        else:
            self._values[self._slot_to_index[slot]] = value

    def __len__(self):
        return len(self._values)

    def __iter__(self):
        if self._slot_to_index is None:
            return ((SlateSlot(a), p) for a, p in enumerate(self._values))
        else:
            return ((s, self._values[i]) for s, i in self._slot_to_index.items())

    @property
    def is_sequence(self):
        return self._slot_to_index is None

    @property
    def slots(self) -> SlateSlots:
        if self._slot_to_index is None:
            return SlateSlots(len(self._values))
        else:
            return SlateSlots(list(self._slot_to_index.keys()))

    @property
    def items(self) -> Sequence[Type]:
        return list(self._values)

    def fill(
        self, values: Sequence[object]
    ) -> Union[Mapping[SlateSlot, object], Sequence[object]]:
        assert len(values) >= len(self._values)
        if self._slot_to_index is None:
            return values[: len(self._values)]
        else:
            return {s: v for s, v in zip(self.slots, values[: len(self._values)])}


# type of valid slate candidates, e.g., doc id
SlateItem = Action


class SlateItems(Items[SlateItem]):
    def _new_item(self, i: int) -> SlateItem:
        return SlateItem(i)


class SlateItemValues(Values[SlateItem]):
    def _new_key(self, k: int) -> SlateItem:
        return SlateItem(k)

    @property
    def items(self) -> SlateItems:
        if self.is_sequence:
            return SlateItems(len(self))
        else:
            return SlateItems(super().items)


class Slate(SlateSlotObjects[SlateItem]):
    """
    Class represents a slate: map from slots to items/docs
    """

    def one_hots(self, items: SlateItems, device=None) -> Tensor:
        oh = torch.zeros((len(self), len(items)), dtype=torch.double, device=device)
        for t, i in zip(oh, self._values):
            t[items.index_of(i)] = 1.0
        return oh

    def slot_values(self, item_values: SlateItemValues) -> SlateSlotValues:
        """
        Map items in the slate to given values
        Args:
            item_values: Map from all items to some values

        Returns:
            List of values in the slate
        """
        if self._slot_to_index is None:
            return SlateSlotValues([item_values[i] for i in self._values])
        else:
            return SlateSlotValues({s: self._values[i] for s, i in self._slot_to_index})

    def __repr__(self):
        return f"{self.__class__.__name__}{{value[{self._values}]}}"


def make_slate(slots: SlateSlots, items: Sequence[SlateItem]) -> Slate:
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
        self._item_size = len(self._values[0])
        for v in self._values[1:]:
            assert self._item_size == len(v)

    def values_tensor(self, device=None) -> Tensor:
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


def make_slot_item_distributions(
    slots: SlateSlots, dists: Sequence[SlateItemValues]
) -> SlateSlotItemExpectations:
    assert len(dists) >= len(slots)
    if slots.is_sequence:
        return SlateSlotItemExpectations(list(dists[: len(slots)]))
    else:
        return SlateSlotItemExpectations(dict(zip(slots, dists[: len(slots)])))


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

    def _new_key(self, k: int) -> SlateItem:
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
            for i1, i2 in zip(items, slate.items):
                if i1 != i2:
                    return 0.0
            return 1.0
        else:
            p = 1.0
            d = 1.0
            for _, i in slate:
                ip = self.probability(i)
                p *= ip / d
                d -= ip
            return Probability(p)

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
                [
                    self.replace(torch.zeros(item_size, dtype=torch.double))
                    for _ in range(len(self))
                ],
            )
            sorted_items, _ = self.sort()
            for item, ds in zip(sorted_items, self._slot_item_expectations.items):
                ds[item] = 1.0
        else:
            self._normalize()
            if (len(slots) < 5 and len(self) < 47) or (
                len(slots) < 6 and len(self) < 19
            ):
                self._calculate_expectations(slots)
            else:
                self._sample_expectations(slots, 20000)
        return self._slot_item_expectations

    def _sample_expectations(self, slots: SlateSlots, num_samples: int):
        slate_size = len(slots)
        item_size = len(self)
        dm = torch.zeros((slate_size, item_size), dtype=torch.double)
        ri = torch.arange(slate_size)
        ws = self._probabilities.repeat((num_samples, 1))
        for _ in range(item_size):
            samples = torch.multinomial(ws, slate_size)
            for sample in samples:
                dm[ri, sample] += 1
        dm /= num_samples * item_size
        self._slot_item_expectations = make_slot_item_distributions(
            slots, [self.replace(vs) for vs in dm]
        )

    def _calculate_expectations(self, slots: SlateSlots):
        slate_size = len(slots)
        item_size = len(self)
        dm = torch.zeros((slate_size, item_size), dtype=torch.double)
        dm[0] = self._probabilities
        buffer = [({}, 1.0, 0.0, 1.0)]
        for d in dm[1:]:
            next_buffer = []
            for b in buffer:
                for i, i_prob in enumerate(self._probabilities):
                    if i in b[0]:
                        continue
                    b1 = b[1] * i_prob / (1.0 - b[2])
                    b2 = b[2] + i_prob
                    b3 = b1 / (1.0 - b2)
                    next_buffer.append(({*b[0], i}, b1, b2, b3))
            for i, i_prob in enumerate(self._probabilities):
                p = 0.0
                for b in next_buffer:
                    if i in b[0]:
                        continue
                    p += b[3] * i_prob
                d[i] = p
            buffer = next_buffer
        self._slot_item_expectations = make_slot_item_distributions(
            slots, [self.replace(vs) for vs in dm]
        )

    def sample_slate(self, slots: SlateSlots) -> Slate:
        slate_size = len(slots)
        if self._greedy:
            items = super().greedy(slate_size)
        else:
            items = super().sample(slate_size)
        if slate_size == 1:
            items = [items]
        return make_slate(slots, items)


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
            if (slate_size < 5 and item_size < 47) or (
                slate_size < 6 and item_size < 19
            ):
                self._calculate_expectations()
            else:
                self._sample_expectations(20000 * item_size)
        return self._slot_item_expectations

    def _sample_expectations(self, num_samples: int):
        slate_size = len(self.slots)
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
        self._values[0]._normalize()
        dm[0] = self._values[0]._probabilities
        buffer = [({}, 1.0, 0.0, 1.0)]
        for d, probs in zip(dm[1:], self._values[1:]):
            next_buffer = []
            for b in buffer:
                for i, i_prob in enumerate(probs):
                    if i in b[0]:
                        continue
                    b1 = b[1] * i_prob / (1.0 - b[2])
                    b2 = b[2] + i_prob
                    b3 = b1 / (1.0 - b2)
                    next_buffer.append(({*b[0], i}, b1, b2, b3))
            for i, i_prob in enumerate(probs):
                p = 0.0
                for b in next_buffer:
                    if i in b[0]:
                        continue
                    p += b[3] * i_prob
                d[i] = p
            buffer = next_buffer
        self._slot_item_expectations = make_slot_item_distributions(
            self.slots, [its.replace(vs) for its, vs in zip(self._values, dm)]
        )

    def sample_slate(self, slots: SlateSlots) -> Slate:
        slate_size = len(slots)
        ps = self.values_tensor()
        items = []
        if self._greedy:
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


SlateQueryType = Union[int, Tuple[int], float, Tuple[float], np.ndarray, Tensor]
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


class SlateMetric(ABC):
    """
    Metric calculator for a slate: weights (dot) rewards
    """

    def __init__(self, device=None):
        self._device = device

    def calculate_reward(
        self,
        slots: SlateSlots,
        rewards: SlateSlotValues = None,
        slot_values: SlateSlotValues = None,
        slot_weights: SlateSlotValues = None,
    ) -> float:
        if slot_values is None:
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
    _weights: Tensor = None

    def _get_discount(self, slate_size: int) -> Tensor:
        if (
            DCGSlateMetric._weights is None
            or DCGSlateMetric._weights.shape[0] < slate_size
            or DCGSlateMetric._weights.device != self._device
        ):
            DCGSlateMetric._weights = torch.reciprocal(
                torch.log2(
                    torch.arange(
                        2, slate_size + 2, dtype=torch.double, device=self._device
                    )
                )
            )
        return DCGSlateMetric._weights[:slate_size]

    def slot_weights(self, slots: SlateSlots) -> SlateSlotValues:
        return slots.fill(self._get_discount(len(slots)))

    def slot_values(self, rewards: SlateSlotValues) -> SlateSlotValues:
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
        return slots.fill(self._get_discount(slate_size) / idcg)


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
    log_slate: Slate
    log_rewards: SlateSlotValues
    slot_probabilities: Optional[SlateSlotValues] = None
    log_slate_probability: float = 0.0
    tgt_slate_probability: float = 0.0

    def validate(self, slate_size: int, item_size: int):
        assert len(self.log_slate) == slate_size
        assert len(self.log_rewards) == slate_size
        assert self.log_slate_probability <= 1.0
        assert self.tgt_slate_probability <= 1.0


@dataclass(frozen=True)
class LogEpisode:
    context: SlateContext
    metric: SlateMetric
    samples: Iterable[LogSample]
    # probability for each item being places at each slot
    _log_slot_item_probabilities: Optional[SlateSlotItemProbabilities] = None
    # item probability distribution from behavior policy
    _log_item_probabilities: Optional[SlateItemProbabilities] = None
    _tgt_slot_item_probabilities: Optional[SlateSlotItemProbabilities] = None
    # item probability distribution from target policy
    _tgt_item_probabilities: Optional[SlateItemProbabilities] = None
    gt_item_rewards: Optional[SlateItemValues] = None

    def validate(self):
        slate_size = len(self.context.slots)
        item_size = len(self.items)
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
        for s in self.samples:
            s.validate(slate_size, item_size)

    def log_slot_item_expectations(
        self, slots: SlateSlots
    ) -> Optional[SlateSlotItemExpectations]:
        if self._log_slot_item_probabilities is not None:
            return self._log_slot_item_probabilities.slot_item_expectations()
        if self._log_item_probabilities is not None:
            return self._log_item_probabilities.slot_item_expectations(slots)
        return None

    def log_slate_probability(self, slate: Slate) -> float:
        if self._log_slot_item_probabilities is not None:
            return self._log_slot_item_probabilities.slate_probability(slate)
        if self._log_item_probabilities is not None:
            return self._log_item_probabilities.slate_probability(slate)
        else:
            return 0.0

    def tgt_slot_expectations(
        self, slots: SlateSlots
    ) -> Optional[SlateSlotItemExpectations]:
        if self._tgt_slot_item_probabilities is not None:
            return self._tgt_slot_item_probabilities.slot_item_expectations()
        if self._tgt_item_probabilities is not None:
            return self._tgt_item_probabilities.slot_item_expectations(slots)
        return None

    def tgt_slate_probability(self, slate: Slate) -> float:
        if self._tgt_slot_item_probabilities is not None:
            return self._tgt_slot_item_probabilities.slate_probability(slate)
        if self._tgt_item_probabilities is not None:
            return self._tgt_item_probabilities.slate_probability(slate)
        else:
            return 0.0

    @property
    def items(self) -> SlateItems:
        if self._log_slot_item_probabilities is not None:
            return self._log_slot_item_probabilities._values[0].items
        if self._log_item_probabilities is not None:
            return self._log_item_probabilities.items
        return SlateItems(0)


@dataclass(frozen=True)
class SlateEstimatorInput:
    episodes: Iterable[LogEpisode]
    tgt_model: SlateModel = None  # target model, used by DM

    def validate(self):
        for e in self.episodes:
            e.validate()


class DMEstimator(Estimator):
    """
    Direct Method estimator
    """

    def evaluate(self, input: SlateEstimatorInput, *kwargs) -> EstimatorResults:
        input.validate()
        if input.tgt_model is None:
            logging.error("Target model is none, DM is not available")
            return self.results
        for episode in input.episodes:
            log_avg = RunningAverage()
            tgt_avg = RunningAverage()
            gt_avg = RunningAverage()
            tgt_slot_expects = episode.tgt_slot_expectations(episode.context.slots)
            if tgt_slot_expects is None:
                logging.warning(f"Target slot expectations not available")
                continue
            gt_slot_rewards = None
            if episode.gt_item_rewards is not None:
                gt_slot_rewards = tgt_slot_expects.expected_rewards(
                    episode.gt_item_rewards
                )
            for sample in episode.samples:
                log_avg.add(episode.metric(episode.context.slots, sample.log_rewards))
                tgt_item_rewards = input.tgt_model.item_rewards(episode.context)
                tgt_slot_rewards = tgt_slot_expects.expected_rewards(tgt_item_rewards)
                tgt_avg.add(episode.metric(episode.context.slots, tgt_slot_rewards))
                if gt_slot_rewards is not None:
                    gt_avg.add(episode.metric(episode.context.slots, gt_slot_rewards))
            self._append_estimate(log_avg.average, tgt_avg.average, gt_avg.average)
        return self.results


class IPSEstimator(Estimator):
    def __init__(
        self, weight_clamper: Clamper = None, weighted: bool = True, device=None
    ):
        super().__init__(device)
        self._weight_clamper = (
            weight_clamper if weight_clamper is not None else Clamper()
        )
        self._weighted = weighted

    def evaluate(self, input: SlateEstimatorInput, *kwargs) -> EstimatorResults:
        input.validate()
        for episode in input.episodes:
            log_avg = RunningAverage()
            tgt_avg = RunningAverage()
            acc_weight = 0.0
            gt_avg = RunningAverage()
            gt_slot_rewards = None
            if episode.gt_item_rewards is not None:
                tgt_slot_expects = episode.tgt_slot_expectations(episode.context.slots)
                if tgt_slot_expects is not None:
                    gt_slot_rewards = tgt_slot_expects.expected_rewards(
                        episode.gt_item_rewards
                    )
            for sample in episode.samples:
                log_prob = sample.log_slate_probability
                if log_prob <= 0.0:
                    log_prob = episode.log_slate_probability(sample.log_slate)
                if log_prob <= 0.0:
                    logging.warning(f"Invalid log slate probability: {log_prob}")
                    continue
                tgt_prob = sample.tgt_slate_probability
                if tgt_prob <= 0.0:
                    tgt_prob = episode.tgt_slate_probability(sample.log_slate)
                if tgt_prob <= 0.0:
                    logging.warning(f"Invalid target probability: {tgt_prob}")
                    continue
                weight = self._weight_clamper(tgt_prob / log_prob)
                log_reward = episode.metric(episode.context.slots, sample.log_rewards)
                log_avg.add(log_reward)
                tgt_avg.add(log_reward * weight)
                acc_weight += weight
                if gt_slot_rewards is not None:
                    gt_avg.add(episode.metric(episode.context.slots, gt_slot_rewards))
            if tgt_avg.count == 0:
                continue
            if self._weighted:
                self._append_estimate(
                    log_avg.average, tgt_avg.total / acc_weight, gt_avg.average
                )
            else:
                self._append_estimate(log_avg.average, tgt_avg.average, gt_avg.average)
        return self.results


class PseudoInverseEstimator(Estimator):
    """
    Estimator from reference 2
    """

    def __init__(
        self, weight_clamper: Clamper = None, weighted: bool = True, device=None
    ):
        super().__init__(device)
        self._weight_clamper = (
            weight_clamper if weight_clamper is not None else Clamper()
        )
        self._weighted = weighted

    def evaluate(self, input: SlateEstimatorInput, *kwargs) -> EstimatorResults:
        input.validate()
        for episode in input.episodes:
            log_avg = RunningAverage()
            tgt_avg = RunningAverage()
            acc_weight = 0.0
            gt_avg = RunningAverage()
            log_slot_expects = episode.log_slot_item_expectations(episode.context.slots)
            if log_slot_expects is None:
                logging.warning(f"Log slot distribution not available")
                continue
            tgt_slot_expects = episode.tgt_slot_expectations(episode.context.slots)
            if tgt_slot_expects is None:
                logging.warning(f"Target slot distribution not available")
                continue
            log_indicator = log_slot_expects.values_tensor(self._device)
            tgt_indicator = tgt_slot_expects.values_tensor(self._device)
            lm = len(episode.context.slots) * len(episode.items)
            gamma = torch.pinverse(
                torch.mm(log_indicator.view((lm, 1)), log_indicator.view((1, lm)))
            )
            gt_slot_rewards = None
            if episode.gt_item_rewards is not None:
                gt_slot_rewards = tgt_slot_expects.expected_rewards(
                    episode.gt_item_rewards
                )
            for sample in episode.samples:
                log_reward = episode.metric(episode.context.slots, sample.log_rewards)
                log_avg.add(log_reward)
                ones = sample.log_slate.one_hots(episode.items, self._device)
                weight = self._weight_clamper(
                    torch.mm(
                        tgt_indicator.view((1, lm)), torch.mm(gamma, ones.view(lm, 1))
                    )
                )
                tgt_avg.add(log_reward * weight)
                acc_weight += weight
                if gt_slot_rewards is not None:
                    gt_avg.add(episode.metric(episode.context.slots, gt_slot_rewards))
            if tgt_avg.count == 0:
                continue
            if self._weighted:
                self._append_estimate(
                    log_avg.average, tgt_avg.total / acc_weight, gt_avg.average
                )
            else:
                self._append_estimate(log_avg.average, tgt_avg.average, gt_avg.average)
        return self.results


class PBMEstimator(Estimator):
    """
    Estimator from reference 1: Position-Based Click Model
    """

    def __init__(
        self, weight_clamper: Clamper = None, weighted: bool = True, device=None
    ):
        super().__init__(device)
        self._weight_clamper = (
            weight_clamper if weight_clamper is not None else Clamper()
        )
        self._weighted = weighted

    def evaluate(self, input: SlateEstimatorInput, *kwargs) -> EstimatorResults:
        input.validate()
        for episode in input.episodes:
            log_avg = RunningAverage()
            tgt_avg = RunningAverage()
            acc_weight = 0.0
            gt_avg = RunningAverage()
            log_slot_expects = episode.log_slot_item_expectations(episode.context.slots)
            if log_slot_expects is None:
                logging.warning(f"Log slot distribution not available")
                continue
            tgt_slot_expects = episode.tgt_slot_expectations(episode.context.slots)
            if tgt_slot_expects is None:
                logging.warning(f"Target slot distribution not available")
                continue
            slate_size = len(episode.context.slots)
            gt_slot_rewards = None
            if episode.gt_item_rewards is not None:
                gt_slot_rewards = tgt_slot_expects.expected_rewards(
                    episode.gt_item_rewards
                )
            for sample in episode.samples:
                slot_weights = episode.metric.slot_weights(episode.context.slots)
                log_reward = episode.metric.calculate_reward(
                    episode.context.slots, sample.log_rewards, None, slot_weights
                )
                log_avg.add(log_reward)
                weights = slot_weights.values.to(device=self._device)
                if sample.slot_probabilities is not None:
                    weights *= sample.slot_probabilities.values
                h = torch.zeros(slate_size, dtype=torch.double, device=self._device)
                p = torch.zeros(slate_size, dtype=torch.double, device=self._device)
                i = 0
                for slot, item in sample.log_slate:
                    h[i] = log_slot_expects[slot][item]
                    p[i] = tgt_slot_expects[slot][item]
                    i += 1
                ips = torch.tensordot(h, weights, dims=([0], [0])) / torch.tensordot(
                    p, weights, dims=([0], [0])
                )
                ips = self._weight_clamper(ips)
                if ips <= 0.0 or math.isinf(ips) or math.isnan(ips):
                    continue
                tgt_avg.add(log_reward * ips)
                acc_weight += ips
                if gt_slot_rewards is not None:
                    gt_avg.add(
                        episode.metric.calculate_reward(
                            episode.context.slots, gt_slot_rewards
                        )
                    )
            if tgt_avg.count == 0:
                continue
            if self._weighted:
                self._append_estimate(
                    log_avg.average, tgt_avg.total / acc_weight, gt_avg.average
                )
            else:
                self._append_estimate(log_avg.average, tgt_avg.average, gt_avg.average)
        return self.results

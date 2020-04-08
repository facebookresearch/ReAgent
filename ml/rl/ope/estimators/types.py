#!/usr/bin/env python3

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import reduce
from typing import (
    Generic,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import torch
from torch import Tensor


def is_array(obj):
    return isinstance(obj, Tensor) or isinstance(obj, np.ndarray)


Type = TypeVar("Type")


@dataclass(frozen=True)
class TypeWrapper(Generic[Type]):
    value: Type

    def __index__(self):
        try:
            return int(self.value)
        except Exception:
            raise ValueError(f"{self} cannot be used as index")

    def __int__(self):
        try:
            return int(self.value)
        except Exception:
            raise ValueError(f"{self} cannot be converted to int")

    def __hash__(self):
        if (
            isinstance(self.value, int)
            or isinstance(self.value, float)
            or isinstance(self.value, tuple)
        ):
            return hash(self.value)
        elif isinstance(self.value, Tensor):
            return hash(tuple(self.value.numpy().flatten()))
        elif isinstance(self.value, np.ndarray):
            return hash(tuple(self.value.flatten()))
        elif isinstance(self.value, list):
            return hash(tuple(self.value))
        else:
            raise TypeError

    def __eq__(self, other):
        if not isinstance(other, TypeWrapper):
            return False
        if isinstance(self.value, Tensor):
            if isinstance(other.value, Tensor):
                return torch.equal(self.value, other.value)
            else:
                raise TypeError(f"{self} cannot be compared with non-tensor")
        elif isinstance(self.value, np.ndarray):
            return np.array_equal(self.value, other.value)
        else:
            return self.value == other.value

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if not isinstance(other, TypeWrapper):
            return False
        if isinstance(self.value, Tensor) and isinstance(other.value, Tensor):
            return torch.lt(self.value, other.value).prod().item()
        elif isinstance(self.value, np.ndarray) and isinstance(other.value, np.ndarray):
            return np.less(self.value, other.value).prod()
        else:
            return self.value < other.value

    def __repr__(self):
        return f"{self.__class__.__name__}{{value[{self.value}]}}"


class Values(Generic[Type], ABC):
    """
    Generic class for a map from item to its value.
    It supports [] indexing, and iterator protocol

    Attributes:
        items: list of items
        values: list of their values
    """

    def __init__(
        self, values: Union[Mapping[Type, float], Sequence[float], np.ndarray, Tensor]
    ):
        self._key_to_index = None
        self._index_to_key = None
        if isinstance(values, Tensor):
            self._values = values.to(dtype=torch.double)
        elif isinstance(values, np.ndarray):
            self._values = torch.as_tensor(values, dtype=torch.double)
        elif isinstance(values, Sequence):
            self._values = torch.tensor(values, dtype=torch.double)
        elif isinstance(values, Mapping):
            self._key_to_index = dict(zip(values.keys(), range(len(values))))
            self._index_to_key = list(values.keys())
            self._values = torch.tensor(list(values.values()), dtype=torch.double)
        else:
            raise TypeError(f"Unsupported values type {type(values)}")
        self._reset()

    def _reset(self):
        self._probabilities = None
        self._is_normalized = False
        self._unzipped = None
        self._sorted = None

    def __getitem__(self, key: Type) -> float:
        if self._key_to_index is not None:
            return self._values[self._key_to_index[key]].item()
        else:
            return self._values[key].item()

    def __setitem__(self, key: Type, value: float):
        if self._key_to_index is not None:
            self._values[self._key_to_index[key]] = value
        else:
            self._values[key] = value
        self._reset()

    @abstractmethod
    def _new_key(self, k: int) -> Type:
        pass

    def __iter__(self):
        if self._key_to_index is not None:
            return ((k, self._values[i]) for k, i in self._key_to_index.items())
        else:
            return ((self._new_key(a), p.item()) for a, p in enumerate(self._values))

    def __len__(self) -> int:
        return self._values.shape[0]

    @property
    def is_sequence(self):
        return self._key_to_index is None

    def sort(self, descending: bool = True) -> Tuple[Sequence[Type], Tensor]:
        """
        Sort based on values

        Args:
            descending: sorting order

        Returns:
            Tuple of sorted indices and values
        """
        if self._sorted is None:
            rs, ids = torch.sort(self._values, descending=descending)
            if self._index_to_key is not None:
                self._sorted = (
                    [self._index_to_key[i.item()] for i in ids],
                    rs.detach(),
                )
            else:
                self._sorted = ([self._new_key(i.item()) for i in ids], rs.detach())
        return self._sorted

    def _unzip(self):
        if self._unzipped is None:
            if self._key_to_index is not None:
                self._unzipped = (
                    list(self._key_to_index.keys()),
                    self._values.clone().detach(),
                )
            else:
                self._unzipped = (
                    [self._new_key(i) for i in range(self._values.shape[0])],
                    self._values.clone().detach(),
                )

    def index_of(self, item: Type) -> int:
        if self._key_to_index is None and isinstance(item.value, int):
            if 0 <= item.value < self._values.shape[0]:
                return item.value
            else:
                raise ValueError(f"{item} is not valid")
        elif self._key_to_index is not None:
            try:
                return self._key_to_index[item]
            except Exception:
                raise ValueError(f"{item} is not valid")
        else:
            raise ValueError(f"{item} is not valid")

    @property
    def items(self) -> Sequence[Type]:
        self._unzip()
        return self._unzipped[0]

    @property
    def values(self) -> Tensor:
        self._unzip()
        return self._unzipped[1]

    def __repr__(self):
        return f"{self.__class__.__name__}{{values[{self._values}]}}"

    def copy(self) -> "Values":
        cp = self.__class__(self._values.clone().detach())
        if self._key_to_index is not None:
            cp._key_to_index = dict(self._key_to_index)
        if self._index_to_key is not None:
            cp._index_to_key = list(self._index_to_key)
        return cp

    def replace(
        self, values: Union[Mapping[Type, float], Sequence[float], Tensor, np.ndarray]
    ) -> "Values":
        """
        Replace current values with new values, and returns the new copy.
        Current Values object is not changed

        Args:
            values: new value

        Returns:
            Values object with new values
        """
        copy = self.copy()
        if isinstance(values, Tensor):
            assert values.shape[0] == copy._values.shape[0]
            copy._values = values.to(dtype=torch.double)
        elif isinstance(values, np.ndarray):
            assert values.shape[0] == copy._values.shape[0]
            copy._values = torch.as_tensor(values, dtype=torch.double)
        elif isinstance(values, Sequence):
            assert len(values) == copy._values.shape[0]
            copy._values = torch.tensor(values, dtype=torch.double)
        elif isinstance(values, Mapping):
            if copy._key_to_index is None:
                for k, v in values.items():
                    copy._values[k] = v
            else:
                for k, v in values.items():
                    copy._values[copy._key_to_index[k]] = v
        else:
            raise TypeError(f"Unsupported values type {type(values)}")
        return copy

    def _normalize(self):
        if self._is_normalized:
            if self._probabilities is None:
                raise ValueError(f"Invalid distribution {type(self._values)}")
            return
        self._is_normalized = True
        self._probabilities = None
        try:
            dist = self._values.detach().clamp(min=0.0)
            dist /= dist.sum()
            self._probabilities = dist
        except ZeroDivisionError:
            pass

    def probability(self, key: Type) -> float:
        self._normalize()
        if self._probabilities is not None:
            if self._key_to_index is not None:
                return self._probabilities[self._key_to_index[key]].item()
            else:
                return self._probabilities[key].item()
        else:
            return 0.0

    def sample(self, size=1) -> Union[Sequence[Type], Type]:
        self._normalize()
        if self._index_to_key is not None:
            l = [
                self._index_to_key[k.item()]
                for k in torch.multinomial(self._probabilities, size)
            ]
        else:
            l = [
                self._new_key(k.item())
                for k in torch.multinomial(self._probabilities, size)
            ]
        if size == 1:
            return l[0]
        else:
            return l

    def greedy(self, size=1) -> Union[Sequence[Type], Type]:
        sorted_keys, _ = self.sort()
        if size == 1:
            return sorted_keys[0]
        else:
            return sorted_keys[:size]


class Items(Generic[Type], ABC):
    """
    List of items
    """

    def __init__(self, items: Union[Sequence[Type], int]):
        if isinstance(items, int):
            assert items > 0
            self._items = [self._new_item(i) for i in range(items)]
            self._reverse_lookup = None
        else:
            self._items = items
            self._reverse_lookup = {v: i for i, v in enumerate(items)}

    def __getitem__(self, i) -> Type:
        return self._items[i]

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __int__(self):
        if self._reverse_lookup is None:
            return len(self._items)
        else:
            return 0

    @abstractmethod
    def _new_item(self, i: int) -> Type:
        pass

    @property
    def is_sequence(self):
        return self._reverse_lookup is None

    def index_of(self, item: Type) -> int:
        if self._reverse_lookup is None and isinstance(item.value, int):
            if 0 <= item.value < len(self._items):
                return item.value
            else:
                raise ValueError(f"{item} is not valid")
        elif self._reverse_lookup is not None:
            try:
                return self._reverse_lookup[item]
            except Exception:
                raise ValueError(f"{item} is not valid")
        else:
            raise ValueError(f"{item} is not valid")

    def _fill(
        self, values: Union[Mapping[Type, float], Sequence[float], np.ndarray, Tensor]
    ) -> Union[Sequence[float], Mapping[Type, float]]:
        if self._reverse_lookup is None:
            if isinstance(values, Mapping):
                ds = []
                for a in self._items:
                    if a in values:
                        ds.append(values[a])
                    else:
                        ds.append(0.0)
                return ds
            else:
                ds = [0.0] * len(self._items)
                l = min(len(self._items), len(values))
                ds[:l] = values[:l]
                return ds
        elif isinstance(values, Mapping):
            ds = {}
            for a in self._items:
                if a in values:
                    ds[a] = values[a]
                else:
                    ds[a] = 0.0
            return ds
        else:
            raise Type(f"{values} not valid type")


# action type
ActionType = Union[int, Tuple[int], float, Tuple[float], np.ndarray, Tensor]
Action = TypeWrapper[ActionType]

Reward = float
Probability = float


# Action distribution: Action -> probability
#  if action can be indexed, the type is either sequence of float or 1-D tensor,
#  with the indices being the action
class ActionDistribution(Values[Action]):
    def _new_key(self, k: int) -> Action:
        return Action(k)


class ActionSpace(Items[Action]):
    def _new_item(self, i: int) -> Action:
        return Action(i)

    @property
    def space(self) -> Sequence[Action]:
        return self._items

    def distribution(
        self, dist: Union[Mapping[Action, float], Sequence[float], np.ndarray, Tensor]
    ) -> ActionDistribution:
        return ActionDistribution(super()._fill(dist))


class Policy(ABC):
    """
    Policy interface
    """

    def __init__(self, action_space: ActionSpace, device=None):
        self._action_space = action_space
        self._device = device

    @abstractmethod
    def _query(self, context) -> Tuple[Action, ActionDistribution]:
        pass

    def __call__(self, context) -> Tuple[Action, ActionDistribution]:
        return self._query(context)

    @property
    def action_space(self):
        return self._action_space

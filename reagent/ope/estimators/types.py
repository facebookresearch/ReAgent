#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import pickle
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Generic, Mapping, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import torch
from torch import Tensor


def is_array(obj) -> bool:
    return isinstance(obj, Tensor) or isinstance(obj, np.ndarray)


Type = TypeVar("Type")
KeyType = TypeVar("KeyType")
ValueType = TypeVar("ValueType")


@dataclass(frozen=True)
class TypeWrapper(Generic[ValueType]):
    value: ValueType

    def __index__(self) -> int:
        try:
            return int(self.value)
        except Exception:
            raise ValueError(f"{self} cannot be used as index")

    def __int__(self) -> int:
        try:
            return int(self.value)
        except Exception:
            raise ValueError(f"{self} cannot be converted to int")

    def __hash__(self) -> int:
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

    def __eq__(self, other) -> bool:
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

    def __ne__(self, other) -> bool:
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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{{value[{self.value}]}}"


class Objects(Generic[KeyType, ValueType], ABC):
    """
    Generic class for a map from item to its value.
    It supports [] indexing, and iterator protocol

    Attributes:
        items: list of items
        values: list of their values
    """

    def __init__(
        self, values: Union[Mapping[KeyType, ValueType], Sequence[ValueType]]
    ) -> None:
        self._key_to_index = None
        self._index_to_key = None
        self._init_values(values)
        self._reset()

    def _init_values(
        self, values: Union[Mapping[KeyType, ValueType], Sequence[ValueType]]
    ) -> None:
        if isinstance(values, Sequence):
            # pyre-fixme[16]: `Objects` has no attribute `_values`.
            self._values = list(values)
        elif isinstance(values, Mapping):
            self._key_to_index = dict(zip(values.keys(), range(len(values))))
            self._index_to_key = list(values.keys())
            self._values = list(values.values())
        else:
            raise TypeError(f"Unsupported values type {type(values)}")

    def _reset(self) -> None:
        self._unzipped = None
        self._keys = None

    def __getitem__(self, key: KeyType) -> ValueType:
        if self._key_to_index is not None:
            # pyre-fixme[16]: `Objects` has no attribute `_values`.
            return self._values[self._key_to_index[key]]
        else:
            return self._values[key]

    def __setitem__(self, key: KeyType, value: ValueType) -> None:
        if self._key_to_index is not None:
            # pyre-fixme[16]: `Objects` has no attribute `_values`.
            self._values[self._key_to_index[key]] = value
        else:
            self._values[key] = value
        self._reset()

    @abstractmethod
    def _to_key(self, k: int) -> KeyType:
        pass

    def _to_value(self, v) -> ValueType:
        return v

    def __iter__(self):
        if self._key_to_index is not None:
            return (
                (k, self._to_value(self._values[i]))
                for k, i in self._key_to_index.items()
            )
        else:
            return (
                (self._to_key(i), self._to_value(v)) for i, v in enumerate(self._values)
            )

    def __len__(self) -> int:
        # pyre-fixme[16]: `Objects` has no attribute `_values`.
        return len(self._values)

    @property
    def is_sequence(self) -> bool:
        return self._key_to_index is None

    @property
    def _values_copy(self) -> Sequence[ValueType]:
        # pyre-fixme[16]: `Objects` has no attribute `_values`.
        return list(self._values)

    def index_of(self, key: KeyType) -> int:
        if self._key_to_index is None:
            try:
                index = int(key)
                if 0 <= index < len(self):
                    return index
                else:
                    raise ValueError(f"{key} is not valid")
            except Exception:
                raise ValueError(f"{key} is not valid")
        elif self._key_to_index is not None:
            try:
                return self._key_to_index[key]
            except Exception:
                raise ValueError(f"{key} is not valid")
        else:
            raise ValueError(f"{key} is not valid")

    @property
    def keys(self) -> Sequence[KeyType]:
        if self._keys is None:
            if self._key_to_index is not None:
                self._keys = list(self._key_to_index.keys())
            else:
                self._keys = [self._to_key(i) for i in range(len(self))]
        return self._keys

    @property
    def values(self):
        return self._values_copy

    def __repr__(self) -> str:
        # pyre-fixme[16]: `Objects` has no attribute `_values`.
        return f"{self.__class__.__name__}{{values[{self._values}]}}"


class Values(Objects[KeyType, float]):
    """
    Generic class for a map from item to its value.
    It supports [] indexing, and iterator protocol

    Attributes:
        items: list of items
        values: list of their values
    """

    def __init__(
        self,
        values: Union[Mapping[KeyType, float], Sequence[float], np.ndarray, Tensor],
    ) -> None:
        # pyre-fixme[6]: Expected `Union[Mapping[Variable[KeyType],
        #  Variable[ValueType]], Sequence[Variable[ValueType]]]` for 1st param but got
        #  `Union[Mapping[Variable[KeyType], float], Sequence[float], Tensor,
        #  np.ndarray]`.
        super().__init__(values)

    def _init_values(
        self,
        values: Union[Mapping[KeyType, float], Sequence[float], np.ndarray, Tensor],
    ) -> None:
        if isinstance(values, Tensor):
            # pyre-fixme[16]: `Values` has no attribute `_values`.
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

    def _reset(self) -> None:
        super()._reset()
        # pyre-fixme[16]: `Values` has no attribute `_probabilities`.
        self._probabilities = None
        # pyre-fixme[16]: `Values` has no attribute `_is_normalized`.
        self._is_normalized = False
        # pyre-fixme[16]: `Values` has no attribute `_sorted`.
        self._sorted = None

    def __getitem__(self, key: KeyType) -> float:
        return super().__getitem__(key).item()

    def _to_value(self, v: Tensor) -> float:
        return v.item()

    def __len__(self) -> int:
        # pyre-fixme[16]: `Values` has no attribute `_values`.
        return self._values.shape[0]

    def sort(self, descending: bool = True) -> Tuple[Sequence[KeyType], Tensor]:
        """
        Sort based on values

        Args:
            descending: sorting order

        Returns:
            Tuple of sorted indices and values
        """
        # pyre-fixme[16]: `Values` has no attribute `_sorted`.
        if self._sorted is None:
            # pyre-fixme[16]: `Values` has no attribute `_values`.
            rs, ids = torch.sort(self._values, descending=descending)
            if self._index_to_key is not None:
                self._sorted = (
                    [self._index_to_key[i.item()] for i in ids],
                    rs.detach(),
                )
            else:
                self._sorted = ([self._to_key(i.item()) for i in ids], rs.detach())
        return self._sorted

    @property
    def _values_copy(self) -> Tensor:
        # pyre-fixme[16]: `Values` has no attribute `_values`.
        return self._values.clone().detach()

    def replace(
        self,
        values: Union[Mapping[ValueType, float], Sequence[float], Tensor, np.ndarray],
    ) -> "Values":
        """
        Replace current values with new values, and returns the new copy.
        Current Values object is not changed

        Args:
            values: new value

        Returns:
            Values object with new values
        """
        copy = deepcopy(self)
        if isinstance(values, Tensor):
            # pyre-fixme[16]: `Values` has no attribute `_values`.
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

    def _normalize(self) -> None:
        # pyre-fixme[16]: `Values` has no attribute `_is_normalized`.
        if self._is_normalized:
            # pyre-fixme[16]: `Values` has no attribute `_probabilities`.
            if self._probabilities is None:
                raise ValueError(f"Invalid distribution {type(self._values)}")
            return
        self._is_normalized = True
        self._probabilities = None
        try:
            # pyre-fixme[16]: `Values` has no attribute `_values`.
            dist = self._values.detach().clamp(min=0.0)
            dist /= dist.sum()
            self._probabilities = dist
        except ZeroDivisionError:
            pass

    def probability(self, key: ValueType) -> float:
        self._normalize()
        # pyre-fixme[16]: `Values` has no attribute `_probabilities`.
        if self._probabilities is not None:
            if self._key_to_index is not None:
                return self._probabilities[self._key_to_index[key]].item()
            else:
                return self._probabilities[key].item()
        else:
            return 0.0

    def sample(self, size: int = 1) -> Sequence[KeyType]:
        self._normalize()
        if self._index_to_key is not None:
            l = [
                self._index_to_key[k.item()]
                # pyre-fixme[16]: `Values` has no attribute `_probabilities`.
                for k in torch.multinomial(self._probabilities, size)
            ]
        else:
            l = [
                self._to_key(k.item())
                for k in torch.multinomial(self._probabilities, size)
            ]
        return l

    def greedy(self, size: int = 1) -> Sequence[KeyType]:
        sorted_keys, _ = self.sort()
        return sorted_keys[:size]


class Items(Generic[ValueType], ABC):
    """
    List of items
    """

    def __init__(self, items: Union[Sequence[ValueType], int]) -> None:
        if isinstance(items, int):
            assert items > 0
            self._items = [self._new_item(i) for i in range(items)]
            self._reverse_lookup = None
        else:
            self._items = items
            self._reverse_lookup = {v: i for i, v in enumerate(items)}

    def __getitem__(self, i) -> ValueType:
        return self._items[i]

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __int__(self) -> int:
        if self._reverse_lookup is None:
            return len(self._items)
        else:
            return 0

    @abstractmethod
    def _new_item(self, i: int) -> ValueType:
        pass

    @property
    def is_sequence(self) -> bool:
        return self._reverse_lookup is None

    def index_of(self, item: ValueType) -> int:
        if self._reverse_lookup is None:
            # pyre-fixme[16]: `ValueType` has no attribute `value`.
            int_val = int(item.value)
            if 0 <= int_val < len(self._items):
                return int_val
            else:
                raise ValueError(f"{item} is not valid")
        elif self._reverse_lookup is not None:
            try:
                return self._reverse_lookup[item]
            except Exception:
                raise ValueError(f"{item} is not valid")
        else:
            raise ValueError(f"{item} is not valid")

    def fill(
        self,
        values: Union[Mapping[ValueType, float], Sequence[float], np.ndarray, Tensor],
    ) -> Union[Sequence[float], Mapping[ValueType, float]]:
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
            ds = {}
            for a in self._items:
                try:
                    ds[a] = values[self._reverse_lookup[a]]
                except Exception:
                    ds[a] = 0.0
            return ds


# action type
ActionType = Union[int, Tuple[int], float, Tuple[float], np.ndarray, Tensor]
Action = TypeWrapper[ActionType]

Reward = float
Probability = float


# Action distribution: Action -> probability
#  if action can be indexed, the type is either sequence of float or 1-D tensor,
#  with the indices being the action
class ActionDistribution(Values[Action]):
    def _to_key(self, k: int) -> Action:
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
        return ActionDistribution(super().fill(dist))


class Policy(ABC):
    """
    Policy interface
    """

    def __init__(self, action_space: ActionSpace, device=None) -> None:
        self._action_space = action_space
        self._device = device

    @abstractmethod
    def _query(self, context) -> Tuple[Action, ActionDistribution]:
        pass

    def __call__(self, context) -> Tuple[Action, ActionDistribution]:
        return self._query(context)

    @property
    def action_space(self) -> ActionSpace:
        return self._action_space


@dataclass(frozen=True)
class TrainingData:
    train_x: Tensor
    train_y: Tensor
    train_weight: Optional[Tensor]
    validation_x: Tensor
    validation_y: Tensor
    validation_weight: Optional[Tensor]


@dataclass(frozen=True)
class PredictResults:
    predictions: Optional[Tensor]  # shape = [num_samples]
    scores: Tensor  # shape = [num_samples]
    probabilities: Optional[Tensor] = None


class Trainer(ABC):
    def __init__(self) -> None:
        self._model = None

    @staticmethod
    def _sample(
        x: Tensor,
        y: Tensor,
        weight: Optional[Tensor] = None,
        num_samples: int = 0,
        fortran_order: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert x.shape[0] == y.shape[0]
        x_na = x.numpy()
        if fortran_order:
            x_na = x_na.reshape(x.shape, order="F")
        y_na = y.numpy()
        w_na = weight.numpy() if weight is not None else None
        if num_samples > 0 and num_samples < x.shape[0]:
            cs = np.random.choice(x.shape[0], num_samples, replace=False)
            x_na = x_na[cs, :]
            y_na = y_na[cs]
            w_na = w_na[cs] if w_na is not None else None
        return x_na, y_na, w_na

    def reset(self) -> None:
        self._model = None

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    @abstractmethod
    def train(self, data: TrainingData, iterations: int = 1, num_samples: int = 0):
        pass

    @abstractmethod
    def predict(self, x: Tensor, device=None) -> PredictResults:
        pass

    @abstractmethod
    def score(self, x: Tensor, y: Tensor, weight: Optional[Tensor] = None) -> float:
        pass

    def save_model(self, file: str) -> None:
        if self._model is None:
            logging.error(f"{self.__class__.__name__}.save_model: _model is None ")
            return
        try:
            with open(file, "wb") as f:
                pickle.dump(self._model, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            logging.error(f"{file} cannot be accessed.")

    def load_model(self, file: str) -> None:
        try:
            logging.info(f"{self.__class__.__name__}.load_model: {file}")
            with open(file, "rb") as f:
                self._model = pickle.load(f)
        except Exception:
            logging.error(f"{file} cannot be read.")

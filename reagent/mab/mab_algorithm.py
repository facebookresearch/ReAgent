from abc import ABC, abstractmethod
from typing import Optional, List, Tuple

import torch
from torch import Tensor


def get_arm_indices(
    ids_of_all_arms: List[str], ids_of_arms_in_batch: List[str]
) -> List[int]:
    arm_idxs = []
    for i in ids_of_arms_in_batch:
        try:
            arm_idxs.append(ids_of_all_arms.index(i))
        except ValueError:
            raise ValueError(f"Unknown arm_id {i}. Known arm ids: {ids_of_all_arms}")
    return arm_idxs


def place_values_at_indices(values: Tensor, idxs: List[int], total_len: int) -> Tensor:
    """
    We place the values provided in `values` at indices provided in idxs. The values at indices
        not included in `idxs` are filled with zeros.
    TODO: maybe replace with sparse-to-dense tensor function?
    Example:
        place_values_at_indices(Tensor([4,5]), [2,0], 4) == Tensor([5, 0, 4, 0])

    Args:
        values (Tensor): The values
        idxs (List[int]): The indices at which the values have to be placed
        total_len (int): Length of the output tensor
    Return:
        The output tensor
    """
    assert len(values) == len(idxs)
    ret = torch.zeros(total_len)
    ret[idxs] = values
    return ret


def reindex_multiple_tensors(
    all_ids: List[str],
    batch_ids: Optional[List[str]],
    value_tensors: Tuple[Tensor, ...],
) -> Tuple[Tensor, ...]:
    """
    Each tensor from value_tensors is ordered by ids from batch_ids. In the output we
        return these tensors reindexed by all_ids, filling in zeros for missing entries.

    Args:
        all_ids (List[str]): The IDs that specify how to order the elements in the output
        batch_ids (Optional[List[str]]): The IDs that specify how the elements are ordered in the input
        value_tensors (Tuple[Tensor]): A tuple of tensors with elements ordered by `batch_ids`
    Return:
        A Tuple of reindexed tensors
    """
    if batch_ids is None or batch_ids == all_ids:
        # the observations are for all arms are already in correct order
        return value_tensors
    else:
        assert len(batch_ids) == len(
            set(batch_ids)
        )  # make sure no duplicates in arm IDs

        # get the indices of the arms
        arm_idxs = get_arm_indices(all_ids, batch_ids)

        # put elements from the batch in the positions specified by `arm_ids` (missing arms will be zero)
        ret = []
        for v in value_tensors:
            ret.append(place_values_at_indices(v, arm_idxs, len(all_ids)))
        return tuple(ret)


def randomized_argmax(x: torch.Tensor) -> int:
    """
    Like argmax, but return a random (uniformly) index of the max element
    This function makes sense only if there are ties for the max element
    """
    if torch.isinf(x).any():
        # if some scores are inf, return the index for one of the infs
        best_indices = torch.nonzero(torch.isinf(x)).squeeze()
    else:
        max_value = torch.max(x)
        best_indices = torch.nonzero(x == max_value).squeeze()
    if best_indices.ndim == 0:
        # if there is a single argmax
        chosen_idx = int(best_indices)
    else:
        chosen_idx = int(
            best_indices[
                torch.multinomial(
                    1.0 / len(best_indices) * torch.ones(len(best_indices)), 1
                )[0]
            ]
        )
    return chosen_idx


class MABAlgo(torch.nn.Module, ABC):
    def __init__(
        self,
        randomize_ties: bool = True,
        min_num_obs_per_arm: int = 1,
        *,
        n_arms: Optional[int] = None,
        arm_ids: Optional[List[str]] = None,
    ):
        super().__init__()
        if n_arms is not None:
            self.arm_ids = list(map(str, range(n_arms)))
            self.n_arms = n_arms
        if arm_ids is not None:
            self.arm_ids = arm_ids
            self.n_arms = len(arm_ids)
        self.min_num_obs_per_arm = min_num_obs_per_arm
        self.total_n_obs_all_arms = 0
        self.total_n_obs_per_arm = torch.zeros(self.n_arms)
        self.total_sum_reward_per_arm = torch.zeros(self.n_arms)
        self.total_sum_reward_squared_per_arm = torch.zeros(self.n_arms)
        self.randomize_ties = randomize_ties

    def add_batch_observations(
        self,
        n_obs_per_arm: Tensor,
        sum_reward_per_arm: Tensor,
        sum_reward_squared_per_arm: Tensor,
        arm_ids: Optional[List[str]] = None,
    ):
        (
            n_obs_per_arm,
            sum_reward_per_arm,
            sum_reward_squared_per_arm,
        ) = reindex_multiple_tensors(
            all_ids=self.arm_ids,
            batch_ids=arm_ids,
            value_tensors=(
                n_obs_per_arm,
                sum_reward_per_arm,
                sum_reward_squared_per_arm,
            ),
        )

        self.total_n_obs_per_arm += n_obs_per_arm
        self.total_sum_reward_per_arm += sum_reward_per_arm
        self.total_sum_reward_squared_per_arm += sum_reward_squared_per_arm
        self.total_n_obs_all_arms += int(n_obs_per_arm.sum().item())

    def add_single_observation(self, arm_id: str, reward: float):
        """
        Add a single observation (arm played, reward) to the bandit

        Args:
            arm_id (int): Which arm was played
            reward (float): Reward renerated by the arm
        """
        assert arm_id in self.arm_ids
        arm_idx = self.arm_ids.index(arm_id)
        self.total_n_obs_per_arm[arm_idx] += 1
        self.total_sum_reward_per_arm[arm_idx] += reward
        self.total_sum_reward_squared_per_arm[arm_idx] += reward ** 2
        self.total_n_obs_all_arms += 1

    def get_action(self) -> str:
        """
        Get the id of the action chosen by the MAB algorithm

        Returns:
            int: The integer ID of the chosen action
        """
        scores = self()  # calling forward() under the hood
        if self.randomize_ties:
            best_idx = randomized_argmax(scores)
        else:
            best_idx = torch.argmax(scores)
        return self.arm_ids[best_idx]

    def reset(self):
        """
        Reset the MAB to the initial (empty) state.
        """
        self.__init__(randomize_ties=self.randomize_ties, arm_ids=self.arm_ids)

    @abstractmethod
    def get_scores(self) -> Tensor:
        pass

    def forward(self):
        # set `inf` scores for arms which don't have the minimum number of observations
        return torch.where(
            self.total_n_obs_per_arm >= self.min_num_obs_per_arm,
            self.get_scores(),
            torch.tensor(torch.inf, dtype=torch.float),
        )

    def get_avg_reward_values(self) -> Tensor:
        return self.total_sum_reward_per_arm / self.total_n_obs_per_arm

    @classmethod
    def get_scores_from_batch(
        cls,
        n_obs_per_arm: Tensor,
        sum_reward_per_arm: Tensor,
        sum_reward_squared_per_arm: Tensor,
    ) -> Tensor:
        """
        A utility method used to create the bandit, feed in a batch of observations and get the scores in one function call

        Args:
            n_obs_per_arm (Tensor): A tensor of counts of per-arm numbers of observations
            sum_reward_per_arm (Tensor): A tensor of sums of rewards for each arm
            sum_reward_squared_per_arm (Tensor): A tensor of sums of squared rewards for each arm

        Returns:
            Tensor: Array of per-arm scores
        """
        n_arms = len(n_obs_per_arm)
        b = cls(n_arms=n_arms)  # pyre-ignore[45]
        b.add_batch_observations(
            n_obs_per_arm, sum_reward_per_arm, sum_reward_squared_per_arm
        )
        return b()

    def __repr__(self):
        t = ", ".join(
            f"{v:.3f} ({int(n)})"
            for v, n in zip(self.get_avg_reward_values(), self.total_n_obs_per_arm)
        )
        return f"{type(self).__name__}({self.n_arms} arms; {t}"


class RandomActionsAlgo(MABAlgo):
    """
    A MAB algorithm which samples actions uniformly at random
    """

    def get_scores(self) -> Tensor:
        return torch.rand(self.n_arms)


class GreedyAlgo(MABAlgo):
    """
    Greedy algorithm, which always chooses the best arm played so far
    Arms that haven't been played yet are given priority by assigning inf score
    Ties are resolved in favor of the arm with the smallest index.
    """

    def get_scores(self) -> Tensor:
        return self.get_avg_reward_values()

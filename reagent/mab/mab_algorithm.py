from abc import ABC, abstractmethod
from typing import Union, Optional, List

import torch
from torch import Tensor


def get_arm_indices(
    ids_of_all_arms: List[Union[str, int]], ids_of_arms_in_batch: List[Union[str, int]]
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

    TODO: maybe replace with sparse vector function?

    Args:
        values (Tensor): The values
        idxs (List[int]): The indices at which the values have to be placed
        total_len (int): Length of the array
    """
    assert len(values) == len(idxs)
    ret = torch.zeros(total_len)
    ret[idxs] = values
    return ret


class MABAlgo(torch.nn.Module, ABC):
    def __init__(
        self,
        *,
        n_arms: Optional[int] = None,
        arm_ids: Optional[List[Union[str, int]]] = None,
    ):
        super().__init__()
        if n_arms is not None:
            self.arm_ids = list(range(n_arms))
            self.n_arms = n_arms
        if arm_ids is not None:
            self.arm_ids = arm_ids
            self.n_arms = len(arm_ids)
        self.total_n_obs_all_arms = 0
        self.total_n_obs_per_arm = torch.zeros(self.n_arms)
        self.total_sum_reward_per_arm = torch.zeros(self.n_arms)
        self.total_sum_reward_squared_per_arm = torch.zeros(self.n_arms)

    def add_batch_observations(
        self,
        n_obs_per_arm: Tensor,
        sum_reward_per_arm: Tensor,
        sum_reward_squared_per_arm: Tensor,
        arm_ids: Optional[List[Union[str, int]]] = None,
    ):
        if arm_ids is None or arm_ids == self.arm_ids:
            # assume that the observations are for all arms in the default order
            arm_ids = self.arm_ids
            arm_idxs = list(range(self.n_arms))
        else:
            assert len(arm_ids) == len(
                set(arm_ids)
            )  # make sure no duplicates in arm IDs

            # get the indices of the arms
            arm_idxs = get_arm_indices(self.arm_ids, arm_ids)

            # put elements from the batch in the positions specified by `arm_ids` (missing arms will be zero)
            n_obs_per_arm = place_values_at_indices(
                n_obs_per_arm, arm_idxs, self.n_arms
            )
            sum_reward_per_arm = place_values_at_indices(
                sum_reward_per_arm, arm_idxs, self.n_arms
            )
            sum_reward_squared_per_arm = place_values_at_indices(
                sum_reward_squared_per_arm, arm_idxs, self.n_arms
            )

        self.total_n_obs_per_arm += n_obs_per_arm
        self.total_sum_reward_per_arm += sum_reward_per_arm
        self.total_sum_reward_squared_per_arm += sum_reward_squared_per_arm
        self.total_n_obs_all_arms += int(n_obs_per_arm.sum())

    def add_single_observation(self, arm_id: int, reward: float):
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

    def get_action(self) -> Union[str, int]:
        """
        Get the id of the action chosen by the MAB algorithm

        Returns:
            int: The integer ID of the chosen action
        """
        scores = self()  # calling forward() under the hood
        return self.arm_ids[torch.argmax(scores)]

    def reset(self):
        """
        Reset the MAB to the initial (empty) state.
        """
        self.__init__(arm_ids=self.arm_ids)

    @abstractmethod
    def forward(self):
        pass

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
        b = cls(n_arms=n_arms)
        b.add_batch_observations(
            n_obs_per_arm, sum_reward_per_arm, sum_reward_squared_per_arm
        )
        return b()


class RandomActionsAlgo(MABAlgo):
    """
    A MAB algorithm which samples actions uniformly at random
    """

    def forward(self) -> Tensor:
        return torch.rand(self.n_arms)


class GreedyAlgo(MABAlgo):
    """
    Greedy algorithm, which always chooses the best arm played so far
    Arms that haven't been played yet are given priority by assigning inf score
    Ties are resolved in favor of the arm with the smallest index.
    """

    def forward(self) -> Tensor:
        return torch.where(
            self.total_n_obs_per_arm > 0,
            self.get_avg_reward_values(),
            torch.tensor(float("inf")),
        )

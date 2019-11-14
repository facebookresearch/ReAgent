#!/usr/bin/env python3

from typing import List, NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F
from ml.rl.test.gym.open_ai_gym_memory_pool import OpenAIGymMemoryPool


_DEFAULT_QUALITY_MEANS = [(-3.0, 0.0)] * 14 + [(0.0, 3.0)] * 6
_DEFAULT_QUALITY_VARIANCES = [1.0] * 20


class DocumentFeature(NamedTuple):
    topic: torch.Tensor
    length: torch.Tensor
    quality: torch.Tensor

    def as_vector(self):
        """
        Convenient function to get single tensor
        """
        return torch.cat(
            (self.topic, self.length.unsqueeze(dim=2), self.quality.unsqueeze(dim=2)),
            dim=2,
        )


class RecSim:
    """
    An environment described in Section 6 of https://arxiv.org/abs/1905.12767
    """

    def __init__(
        self,
        num_topics: int = 20,
        doc_length: float = 4,
        quality_means: List[Tuple[float, float]] = _DEFAULT_QUALITY_MEANS,
        quality_variances: List[float] = _DEFAULT_QUALITY_VARIANCES,
        initial_budget: float = 200,
        alpha: float = 1.0,
        m: int = 10,
        k: int = 3,
        num_users: int = 5000,
        y: float = 0.3,
        device: str = "cpu",
        seed: int = 2147483647,
    ):
        self.seed = seed
        self.device = torch.device(device)
        self.num_users = num_users
        self.num_topics = num_topics
        self.initial_budget = initial_budget
        self.reset()

        self.doc_length = doc_length
        self.alpha = alpha
        self.m = m
        self.k = k
        self.y = y
        self.p_d = torch.ones(self.num_topics, device=self.device) / self.num_topics
        assert (
            len(quality_variances) == len(quality_means) == num_topics
        ), f"Expecting {num_topics}: got {quality_means} and {quality_variances}"
        mean_ranges = torch.tensor(quality_means, device=self.device)

        self.quality_means = (
            torch.rand(num_topics, device=self.device, generator=self.generator)
            * (mean_ranges[:, 1] - mean_ranges[:, 0])
            + mean_ranges[:, 0]
        )
        self.quality_variances = torch.tensor(quality_variances, device=self.device)

    def reset(self) -> None:
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(self.seed)
        self.users = self.sample_users(self.num_users)
        self.active_user_ids = torch.arange(
            start=1, end=self.num_users + 1, device=self.device, dtype=torch.long
        )
        self.user_budgets = torch.full(
            (self.num_users,), self.initial_budget, device=self.device
        )
        self.candidates = None

    def obs(self) -> Tuple[torch.Tensor, torch.Tensor, DocumentFeature]:
        """
        Agent can observe:
        - User interest vector
        - Document topic vector
        - Document length
        - Document quality
        """
        if self.candidates is None:
            self.candidates = self.sample_documents(len(self.active_user_ids))
        return self.active_user_ids, self.users, self.candidates

    def step(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        assert self.candidates is not None
        slate = self.select(self.candidates, action, True)
        user_choice, interest = self.compute_user_choice(slate)
        selected_choice = self.select(slate, user_choice, False)
        self.update_user_interest(selected_choice)
        self.update_user_budget(selected_choice)
        num_alive_sessions = self.update_active_users()
        self.candidates = None
        # TODO: Figure out what was the reward in the paper
        # Here, the reward is the length of selected video
        reward = selected_choice.length * (user_choice != action.shape[1])
        return reward, user_choice, interest, num_alive_sessions

    def select(
        self, candidates: DocumentFeature, indices: torch.Tensor, add_null: bool
    ) -> DocumentFeature:
        batch_size = candidates.topic.shape[0]
        num_candidate = candidates.topic.shape[1]
        num_select = indices.shape[1]
        offset = torch.arange(
            0,
            batch_size * num_candidate,
            step=num_candidate,
            dtype=torch.long,
            device=self.device,
        ).repeat_interleave(num_select)
        select_indices = indices.view(-1) + offset

        topic = candidates.topic.view(-1, self.num_topics)[select_indices].view(
            batch_size, num_select, self.num_topics
        )
        length = candidates.length.view(-1)[select_indices].view(batch_size, num_select)
        quality = candidates.quality.view(-1)[select_indices].view(
            batch_size, num_select
        )

        if add_null:
            length = torch.cat(
                (length, length.new_full((batch_size, 1), self.doc_length)), dim=1
            )
            topic = torch.cat(
                (topic, topic.new_zeros(batch_size, 1, self.num_topics)), dim=1
            )
            quality = torch.cat((quality, quality.new_zeros(batch_size, 1)), dim=1)
        return DocumentFeature(topic=topic, length=length, quality=quality)

    def compute_user_choice(
        self, slate: DocumentFeature
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        interest = self.interest(self.users, slate.topic)
        user_choice = torch.multinomial(interest.exp(), 1, generator=self.generator)
        return user_choice, interest

    def update_user_interest(self, selected_choice):
        pos_prob = (self.interest(self.users, selected_choice.topic) + 1) / 2
        positive_mask = torch.bernoulli(pos_prob, generator=self.generator).bool()
        sign = torch.where(
            positive_mask,
            torch.full_like(pos_prob, 1.0),
            torch.full_like(pos_prob, -1.0),
        )
        interest = self.users * selected_choice.topic.view(
            self.users.shape[0], self.num_topics
        )
        delta_interest = (-self.y * interest.abs() + self.y) * (-interest)
        self.users += sign * delta_interest

    def update_user_budget(self, selected_choice):
        bonus = self.bonus(
            self.users,
            selected_choice.topic,
            selected_choice.length,
            selected_choice.quality,
        )
        self.user_budgets -= (selected_choice.length - bonus).view(-1)

    def bonus(self, u, d, length, quality):
        assert (
            length.shape == quality.shape
        ), f"Unexpected shape length: {length.shape} quality: {quality}"
        return 0.9 / 3.4 * length * self.satisfactory(u, d, quality)

    def update_active_users(self) -> int:
        alive_indices = (self.user_budgets > 0.0).nonzero().squeeze(1)
        if alive_indices.shape[0] < self.user_budgets.shape[0]:
            self.user_budgets = self.user_budgets[alive_indices]
            self.users = self.users[alive_indices]
            self.active_user_ids = self.active_user_ids[alive_indices]
        return alive_indices.shape[0]

    def interest(self, u, d):
        """
        Args:
          u: shape [batch, T]
          d: shape [batch, k, T]
        """
        assert (
            u.dim() == 2
            and d.dim() == 3
            and u.shape[0] == d.shape[0]
            and u.shape[1] == d.shape[2]
        ), f"Shape mismatch u: {u.shape}, d: {d.shape}"
        return torch.bmm(u.unsqueeze(1), d.transpose(1, 2)).squeeze(1)

    def satisfactory(self, u, d, quality):
        assert (
            u.dim() == 2
            and d.dim() == 3
            and quality.dim() == 2
            and u.shape[0] == d.shape[0] == quality.shape[0]
            and d.shape[1] == quality.shape[1]
        ), f"Shape mismatch u: {u.shape}, d: {d.shape}, quality: {quality.shape}"
        if self.alpha == 1.0:
            return quality
        return (1 - self.alpha) * self.interest(u, d) + self.alpha * quality

    def sample_users(self, n):
        """
        User is represented by vector of topic interest, uniformly sampled from [-1, 1]
        """
        return torch.rand((n, self.num_topics), generator=self.generator) * 2 - 1

    def sample_documents(self, n: int) -> DocumentFeature:
        num_docs = n * self.m
        topics = torch.multinomial(
            self.p_d, num_docs, replacement=True, generator=self.generator
        )
        means = self.quality_means[topics]
        variances = self.quality_variances[topics]
        quality = torch.normal(means, variances, generator=self.generator).view(
            n, self.m
        )
        embedding = (
            F.one_hot(topics, self.num_topics).view(n, self.m, self.num_topics).float()
        )
        length = torch.full((n, self.m), self.doc_length, device=self.device)
        return DocumentFeature(topic=embedding, quality=quality, length=length)

    def rollout_policy(
        self, policy, memory_pool: Optional[OpenAIGymMemoryPool] = None
    ) -> float:
        prev_obs = None
        prev_action = None
        prev_user_choice = None
        prev_reward = None
        prev_interest = None

        policy_reward = 0

        while True:
            obs = self.obs()
            active_user_idxs, user_features, candidate_features = obs

            item_idxs = policy(obs, self)
            reward, user_choice, interest, num_alive = self.step(item_idxs)

            policy_reward += reward.sum().item()

            action_features = self.select(
                candidate_features, item_idxs, True
            ).as_vector()

            if memory_pool is not None and prev_obs is not None:
                prev_active_user_idxs, prev_user_features, prev_candidate_features = (
                    prev_obs
                )
                i, j = 0, 0
                while i < len(prev_active_user_idxs):
                    mdp_id = prev_active_user_idxs[i]
                    state = prev_user_features[i]
                    possible_actions = prev_action[i]
                    action = possible_actions[prev_user_choice[i]].view(-1)
                    possible_actions_mask = torch.ones(self.k + 1, dtype=torch.uint8)
                    # HACK: Since reward is going to be masked, this is OK
                    item_reward = prev_reward[i].repeat(self.k + 1)
                    reward_mask = torch.arange(self.k + 1) == prev_user_choice[i]
                    propensity = F.softmax(prev_interest[i], dim=0)

                    if j < len(active_user_idxs) and mdp_id == active_user_idxs[j]:
                        # not terminated
                        terminal = False
                        next_state = user_features[j]
                        possible_next_actions = action_features[j]
                        next_action = possible_next_actions[user_choice[j]].view(-1)
                        next_propensity = F.softmax(interest[j], dim=0)
                        j += 1
                    else:
                        terminal = True
                        next_state = torch.zeros_like(state)
                        possible_next_actions = torch.zeros_like(action)
                        next_action = possible_next_actions[0].view(-1)
                        next_propensity = torch.zeros_like(propensity)

                    # This doesn't matter
                    possible_next_actions_mask = torch.ones(
                        self.k + 1, dtype=torch.uint8
                    )

                    memory_pool.insert_into_memory(
                        state=state,
                        action=action,
                        reward=item_reward,
                        next_state=next_state,
                        next_action=next_action,
                        terminal=terminal,
                        possible_next_actions=possible_next_actions,
                        possible_next_actions_mask=possible_next_actions_mask,
                        time_diff=1.0,
                        possible_actions=possible_actions,
                        possible_actions_mask=possible_actions_mask,
                        policy_id=1,
                        propensity=propensity,
                        next_propensity=next_propensity,
                        reward_mask=reward_mask,
                    )

                    i += 1

            prev_obs = obs
            prev_action = action_features
            prev_user_choice = user_choice
            prev_reward = reward
            prev_interest = interest

            if num_alive == 0:
                break

        return policy_reward

#!/usr/bin/env python3

from typing import List, NamedTuple, Tuple

import torch
import torch.nn.functional as F


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

    def reset(self):
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

    def obs(self):
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

    def step(self, action: torch.Tensor) -> int:
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

    def select(self, candidates, indices, add_null):
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

    def compute_user_choice(self, slate):
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

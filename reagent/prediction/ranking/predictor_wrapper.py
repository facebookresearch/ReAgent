from typing import Tuple, List, Optional

import torch
import torch.nn.functional as F


class DeterminantalPointProcessPredictorWrapper(torch.jit.ScriptModule):
    """http://jgillenw.com/cikm2018.pdf Algorithm 1"""

    def __init__(self, alpha, rerank_topk: Optional[int] = None) -> None:
        super().__init__()
        # control the strength of encouragement for diversity
        self.alpha = alpha
        # hard code this value so jit.script can work
        self.MIN_VALUE = -3.4e38
        # if None, will rerank the full slate
        self.rerank_topk = rerank_topk
        if self.rerank_topk is not None:
            # pyre-fixme[58]: `>` is not supported for operand types `Optional[int]`
            #  and `int`.
            assert self.rerank_topk > 0

    def unchosen_dets(self, L, chosen: List[int]):
        slate_size = L.shape[0]
        dets = torch.full((slate_size,), self.MIN_VALUE, device=L.device)
        for i in range(slate_size):
            if i not in chosen:
                dets[i] = torch.det(L[:, chosen + [i]][chosen + [i]])
        return dets

    def greedy_select(self, L):
        slate_size = L.shape[0]
        dets = torch.zeros(slate_size, slate_size, device=L.device)
        chosen: List[int] = []
        unchosen = torch.ones(slate_size)

        if self.rerank_topk is not None:
            rerank_topk = min(self.rerank_topk, slate_size)
        else:
            rerank_topk = slate_size

        for i in range(rerank_topk):
            unchosen_dets = self.unchosen_dets(L, chosen)
            dets[i, :] = unchosen_dets
            chosen_idx = torch.argmax(unchosen_dets)
            chosen.append(chosen_idx.item())
            unchosen[chosen_idx] = 0

        final_order = torch.tensor(chosen)
        if rerank_topk != slate_size:
            final_order = torch.cat((final_order, torch.nonzero(unchosen).flatten()))

        return final_order, dets

    @torch.jit.script_method
    def forward(
        self,
        quality_scores: torch.Tensor,
        feature_vectors: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # quality_scores shape: num_items, 1
        # feature_vectors shape: num_items, num_feat
        quality_scores = quality_scores.float()
        feature_vectors = F.normalize(feature_vectors.float(), p=2.0, dim=1)

        num_items = quality_scores.shape[0]
        B = (self.alpha ** 0.5) * quality_scores * feature_vectors
        # pyre-fixme[16]: `Tensor` has no attribute `T`.
        L = torch.mm(B, B.T)
        L[torch.arange(num_items), torch.arange(num_items)] = (
            quality_scores.squeeze(1) ** 2
        )
        chosen, dets = self.greedy_select(L)

        return chosen, dets, L, B

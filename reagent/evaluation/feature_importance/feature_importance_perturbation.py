#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import copy
import logging
from collections import defaultdict
from typing import Callable, Any, Optional

import pandas as pd
import torch
import torch.nn as nn
from reagent.core.dataclasses import dataclass
from reagent.evaluation.feature_importance.feature_importance_base import (
    FeatureImportanceBase,
)

logger = logging.getLogger(__name__)


@dataclass
class FeatureImportancePerturbation(FeatureImportanceBase):
    data_loader: Any

    # Consume model (first arg) and data (second arg) to make model predictions
    # Expected to return a tensor of shape (batch_size, 1)
    pred_fn: Callable[[nn.Module, Any], torch.Tensor]

    # Perturb data (first arg) on a specific feature id (second arg)
    perturb_fn: Callable[[Any, int], Any]

    # How many rounds of perturbations for collecting feature importance for each batch
    # The higher it is, the less variance the result will have
    repeat: int = 1

    def compute_feature_importance(self) -> pd.DataFrame:
        feature_importance_vals = defaultdict(list)
        for batch_idx, data in enumerate(self.data_loader):
            for r in range(self.repeat):
                pred_value = self.pred_fn(self.model, data)
                for feature_idx, feature_id in enumerate(self.sorted_feature_ids):
                    copy_data = copy.deepcopy(data)
                    perturbed_data = self.perturb_fn(copy_data, feature_idx)
                    perturbed_pred_value = self.pred_fn(self.model, perturbed_data)
                    feature_importance_vals[feature_id].append(
                        torch.mean(torch.abs(perturbed_pred_value - pred_value))
                    )
                logger.info(f"Processed {batch_idx} batches {r}-th time")

        feature_importance_mean = {
            k: torch.mean(torch.stack(v)).item()
            for k, v in feature_importance_vals.items()
        }
        result_df = pd.DataFrame.from_dict(
            feature_importance_mean, orient="index", columns=["feature_importance"]
        ).sort_values(by=["feature_importance"], ascending=False)
        # Fblearner UI can't show row names (index). So manually add names as a column
        result_df.insert(0, "feature_id", result_df.index)
        return result_df


def create_default_perturb_fn(key: str):
    def default_perturb_fn(
        data,
        feature_idx,
    ):
        val_data, presence_data = data[key]
        batch_size = val_data.shape[0]
        random_idx = torch.randperm(batch_size)
        val_data[:, feature_idx] = val_data[:, feature_idx][random_idx]
        presence_data[:, feature_idx] = presence_data[:, feature_idx][random_idx]
        return data

    return default_perturb_fn

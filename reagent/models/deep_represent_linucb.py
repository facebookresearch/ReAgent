# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import Dict, List, Optional

import torch
from reagent.models.fully_connected_network import FullyConnectedNetwork
from reagent.models.linear_regression import batch_quadratic_form, LinearRegressionUCB
from torch import nn

logger = logging.getLogger(__name__)


class DeepRepresentLinearRegressionUCB(LinearRegressionUCB):
    """
    This is a multi-layer regression model that outputs UCB score.
    There are two modules in this model: MLP module and LinUCB module.
    The MLP module consists of bottom layers whic are trainable by torch optimizer().
    The LinUCB module is the last layer and it is not updated by optimizer but by matrix computations.
    MLP module Refer to paper https://arxiv.org/pdf/2012.01780.pdf.
    LinUCB module refer to paper https://arxiv.org/pdf/2012.01780.pdf.

    The reason to use matrix computations to update model parameters is to output uncertainty besides prediction.
    Example :
        Features(dim=9) --> deep_represent_layers --> Features(dim=3) --> LinUCB --> ucb score

        DeepRepresentLinUCBTrainer(
        (scorer): DeepRepresentLinearRegressionUCB(
            (deep_represent_layers): FullyConnectedNetwork(
            (dnn): Sequential(
                (0): Linear(in_features=9, out_features=6, bias=True)
                (1): ReLU()
                (2): Linear(in_features=6, out_features=3, bias=True)
                (3): Identity()
            )
            )
        )
        (loss_fn): MSELoss()
        )

    In this implementation,
    - ucb = pred_label + uncertainty
    - uncertainty = ucb_alpha * pred_sigma

    - pred_label is the predicted label.
    - pred_sigma reflects the variance associated with the predicted label.
    - ucb_alpha controls the balance between exploration and exploitation,
        - If ucb_alpha is not 0, uncertainty(ucb_alpha * pred_sigma) will be included in the final output.
        - If ucb_alpha is 0, uncertainty(ucb_alpha * pred_sigma) will not be included. The model only outputs a predicted label like a classical supervised model.
    - mlp_out_with_ones is the output from the deep_represent module (with extra column of ones appended for the bias term), also it is the input to the LinUCB module.
    - coefs serve as the top layer LinUCB module in this implementation.
        - it is crutial that coefs will not be updated by gradient back propagation.
        - coefs is defined by @property decorator in LinearRegressionUCB.

    Note in the current implementation the LinUCB coefficients are automatically re-computed at every training step.
    This can be costly for high-dimension LinUCB input (which is the output of `deep_represent_layers`),
    Thus, it's recommended to keep sizes[-1]) low.
    """

    def __init__(
        self,
        input_dim: int,  # MLP input dimension
        sizes: List[int],  # MLP hidden layers of the deep_represent module
        activations: List[str],
        *,
        l2_reg_lambda: float = 1.0,
        ucb_alpha: float = 1.0,
        gamma: float = 1.0,
        use_batch_norm: bool = True,
        dropout_ratio: float = 0.0,
        use_layer_norm: bool = False,
        use_skip_connections: bool = True,
        mlp_layers: Optional[nn.Module] = None,
        nn_e2e: bool = True,
        # nn_e2e=True allows MLP to be trained with a nn.Linear module rather than with LinUCB module. Here the nn.Linear module is used to predict mu, but LinUCB is still used for sigma
    ):
        super().__init__(
            input_dim=sizes[-1]
            + 1,  # self.input_dim is the LinUCB input dimension (equal to MLP output dimension). Adding 1 for bias/intercept
            l2_reg_lambda=l2_reg_lambda,
            ucb_alpha=ucb_alpha,
            gamma=gamma,
        )

        assert input_dim > 0, "input_dim must be > 0, got {}".format(input_dim)
        assert sizes[-1] > 0, "Last layer size must be > 0, got {}".format(sizes[-1])
        assert len(sizes) == len(
            activations
        ), "The numbers of sizes and activations must match; got {} vs {}".format(
            len(sizes), len(activations)
        )

        self.nn_e2e = nn_e2e
        self.raw_input_dim = input_dim  # input to the MLP

        self.linear_layer = nn.Linear(
            in_features=sizes[-1] + 1, out_features=1, bias=False
        )

        # self.raw_input_dim --> MLP --> self.input_dim --> LinUCB --> ucb score
        if mlp_layers is None:
            self.deep_represent_layers = FullyConnectedNetwork(
                [self.raw_input_dim] + sizes,
                activations,
                use_batch_norm=use_batch_norm,
                dropout_ratio=dropout_ratio,
                normalize_output=True,  # output of FullyConnectedNetwork is normalized before fed to LinUCB module
                use_layer_norm=use_layer_norm,
                use_skip_connections=use_skip_connections,
            )
        else:
            self.deep_represent_layers = mlp_layers  # use customized layer

    def input_prototype(self) -> torch.Tensor:
        return torch.randn(1, self.raw_input_dim)

    def forward(
        self, inp: torch.Tensor, ucb_alpha: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Pass raw input to mlp.
        This mlp is trainable to optimizer, i.e., will be updated by torch optimizer(),
            then output of mlp is passed to a LinUCB layer.
        """

        mlp_out = self.deep_represent_layers(inp)
        # preprocess by DeepRepresent module before fed to LinUCB layer
        # trainer needs pred_label and mlp_out to update parameters

        # append a column of ones to features for LinUCB intercept/bias
        mlp_out_with_ones = torch.cat(
            (
                torch.ones(mlp_out.shape[:-1], device=mlp_out.device).unsqueeze(-1),
                mlp_out,
            ),
            -1,
        )

        if ucb_alpha is None:
            ucb_alpha = self.ucb_alpha

        if not self.nn_e2e:
            pred_label = torch.matmul(mlp_out_with_ones, self.coefs)
        else:
            pred_label = self.linear_layer(mlp_out_with_ones).squeeze(-1)

        if ucb_alpha != 0:
            pred_sigma = torch.sqrt(
                batch_quadratic_form(mlp_out_with_ones, self.inv_avg_A)
                / torch.clamp(self.sum_weight, min=0.00001)
            )
            ucb = pred_label + ucb_alpha * pred_sigma
        else:
            pred_sigma = torch.zeros_like(pred_label)
            ucb = pred_label
        return {
            "pred_label": pred_label,
            "pred_sigma": pred_sigma,
            "ucb": ucb,
            "mlp_out_with_ones": mlp_out_with_ones,
        }

    def forward_inference(
        self, inp: torch.Tensor, ucb_alpha: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        B: batch size
        F: raw_input_dim to MLP
        F': input_dim to LinUCB
        Shapes:
            inp: B,F
            pred_label: B
            pred_sigma: B
            ucb: B
            mlp_out_with_ones: B,F'
        """
        mlp_out = self.deep_represent_layers(inp)
        # append a column of ones for LinUCB intercept
        mlp_out_with_ones = torch.cat(
            (
                torch.ones(mlp_out.shape[:-1], device=mlp_out.device).unsqueeze(-1),
                mlp_out,
            ),
            -1,
        )

        model_output = super().forward_inference(
            inp=mlp_out_with_ones, ucb_alpha=ucb_alpha
        )

        pred_label = model_output["pred_label"]
        pred_sigma = model_output["pred_sigma"]
        ucb = model_output["ucb"]
        return {
            "pred_label": pred_label,
            "pred_sigma": pred_sigma,
            "ucb": ucb,
            "mlp_out_with_ones": mlp_out_with_ones,
        }

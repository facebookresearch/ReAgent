#!/usr/bin/env python3

import logging
import math
import time
from typing import Optional

import numpy as np
import torch
from reagent.ope.estimators.types import PredictResults, Trainer, TrainingData
from sklearn.linear_model import Lasso, LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from torch import Tensor


class LinearTrainer(Trainer):
    def __init__(self, is_classifier: bool = False):
        super().__init__()
        self._is_classifier = is_classifier

    def predict(self, x: Tensor, device=None) -> PredictResults:
        if self._model is not None:
            if hasattr(self._model, "predict_proba"):
                proba = torch.as_tensor(
                    self._model.predict_proba(x), dtype=torch.float, device=device
                )
                score = (proba * torch.arange(proba.shape[1])).sum(dim=1)
                return PredictResults(torch.argmax(proba, 1), score, proba)
            elif hasattr(self._model, "predict"):
                return PredictResults(
                    None,
                    torch.as_tensor(
                        self._model.predict(x), dtype=torch.float, device=device
                    ),
                    None,
                )
            else:
                raise AttributeError("model doesn't have predict_proba or predict")
        else:
            raise Exception("model not trained")

    def _score(self, y_true: np.ndarray, y_pred: np.ndarray, weight=None) -> float:
        if self._is_classifier:
            return accuracy_score(y_true, y_pred, sample_weight=weight)
        else:
            return 1.0 / math.pow(
                2, mean_squared_error(y_true, y_pred, sample_weight=weight)
            )

    def score(self, x: Tensor, y: Tensor, weight: Optional[Tensor] = None) -> float:
        y_pred = self._model.predict(x)
        w = weight.numpy() if weight is not None else None
        return self._score(y.numpy(), y_pred, weight=w)


class LassoTrainer(LinearTrainer):
    @property
    def name(self) -> str:
        return "lasso"

    def train(self, data: TrainingData, iterations: int = 1, num_samples: int = 0):
        logging.info("LassoTrainer.train...")
        self._model = None
        best_score = float("-inf")
        for _ in range(iterations):
            x, y, _ = super()._sample(
                data.train_x, data.train_y, data.train_weight, num_samples, True
            )
            sx, sy, ssw = super()._sample(
                data.validation_x, data.validation_y, data.validation_weight
            )
            for alpha in np.logspace(-4, 2, num=7, base=10):
                model = Lasso(
                    alpha=alpha,
                    fit_intercept=False,
                    copy_X=True,
                    max_iter=10000,
                    warm_start=False,
                    selection="random",
                )
                model.fit(x, y)
                y_pred = model.predict(sx)
                score = self._score(sy, y_pred, weight=ssw)
                # score = model.score(sx, sy, ssw)
                logging.info(f"  alpha: {alpha}, score: {score}")
                if score > best_score:
                    best_score = score
                    self._model = model


class DecisionTreeTrainer(LinearTrainer):
    @property
    def name(self) -> str:
        return "decision_tree"

    def train(self, data: TrainingData, iterations: int = 1, num_samples: int = 0):
        logging.info("DecisionTreeTrainer.train...")
        self._model = None
        best_score = float("-inf")
        for _ in range(iterations):
            x, y, sw = super()._sample(
                data.train_x, data.train_y, data.train_weight, num_samples, True
            )
            sx, sy, ssw = super()._sample(
                data.validation_x, data.validation_y, data.validation_weight
            )
            if self._model is None:
                self._model = DecisionTreeRegressor(
                    criterion="mse",
                    splitter="random",
                    max_depth=None,
                    min_samples_split=4,
                    min_samples_leaf=4,
                )
                self._model.fit(x, y, sw)
                y_pred = self._model.predict(sx)
                best_score = self._score(sy, y_pred, weight=ssw)
                logging.info(f"  max_depth: None, score: {best_score}")
            for depth in range(3, 21, 3):
                model = DecisionTreeRegressor(
                    criterion="mse",
                    splitter="random",
                    max_depth=depth,
                    min_samples_split=4,
                    min_samples_leaf=4,
                )
                model.fit(x, y, sw)
                y_pred = model.predict(sx)
                score = self._score(sy, y_pred, weight=ssw)
                # score = model.score(sx, sy, ssw)
                logging.info(f"  max_depth: {depth}, score: {score}")
                if score > best_score:
                    best_score = score
                    self._model = model


class DecisionTreeClassifierTrainer(LinearTrainer):
    def __init__(self):
        super().__init__(True)

    @property
    def name(self) -> str:
        return "decision_tree_classifier"

    def train(self, data: TrainingData, iterations: int = 1, num_samples: int = 0):
        logging.info("DecisionTreeClassifierTrainer.train...")
        self._model = None
        best_score = float("-inf")
        for _ in range(iterations):
            x, y, sw = super()._sample(
                data.train_x, data.train_y, data.train_weight, num_samples, True
            )
            sx, sy, ssw = super()._sample(
                data.validation_x, data.validation_y, data.validation_weight
            )
            for depth in range(3, 21, 3):
                model = DecisionTreeClassifier(
                    criterion="entropy",
                    splitter="random",
                    max_depth=depth,
                    min_samples_split=4,
                    min_samples_leaf=4,
                )
                model.fit(x, y, sw)
                score = model.score(sx, sy, ssw)
                logging.info(f"  max_depth: {depth}, score: {score}")
                if score > best_score:
                    best_score = score
                    self._model = model


class LogisticRegressionTrainer(LinearTrainer):
    def __init__(self, solver: str = "lbfgs"):
        super().__init__(True)
        self._solver = solver

    @property
    def name(self) -> str:
        return "logistic_regression"

    def train(self, data: TrainingData, iterations: int = 1, num_samples: int = 0):
        logging.info("LogisticRegressionTrainer.train...")
        self._model = None
        best_score = float("-inf")
        for _ in range(iterations):
            x, y, sw = super()._sample(
                data.train_x, data.train_y, data.train_weight, num_samples, True
            )
            sx, sy, ssw = super()._sample(
                data.validation_x, data.validation_y, data.validation_weight
            )
            for c in np.logspace(-5, 4, num=10, base=10):
                model = LogisticRegression(
                    C=c,
                    fit_intercept=False,
                    solver=self._solver,
                    max_iter=1000,
                    multi_class="auto",
                    n_jobs=-1,
                )
                model.fit(x, y, sw)
                score = model.score(sx, sy, ssw)
                logging.info(f"  C: {c}, score: {score}")
                if score > best_score:
                    best_score = score
                    self._model = model


class SGDClassifierTrainer(LinearTrainer):
    def __init__(self, loss: str = "log", max_iter: int = 1000):
        super().__init__(True)
        self._loss = loss
        self._max_iter = max_iter

    @property
    def name(self) -> str:
        return "sgd_classifier"

    def train(self, data: TrainingData, iterations: int = 1, num_samples: int = 0):
        logging.info("SGDClassifierTrainer.train...")
        self._model = None
        best_score = float("-inf")
        for _ in range(iterations):
            x, y, _ = super()._sample(
                data.train_x, data.train_y, data.train_weight, num_samples, True
            )
            sx, sy, ssw = super()._sample(
                data.validation_x, data.validation_y, data.validation_weight
            )
            for alpha in np.logspace(-8, -1, num=8, base=10):
                model = SGDClassifier(
                    loss=self._loss,
                    alpha=alpha,
                    random_state=0,
                    max_iter=self._max_iter,
                )
                model.fit(x, y)
                score = model.score(sx, sy, ssw)
                logging.info(f"  alpha: {alpha}, score: {score}")
                if score > best_score:
                    best_score = score
                    self._model = model


class LinearNet(torch.nn.Module):
    def __init__(
        self,
        D_in: int,
        H: int,
        D_out: int,
        hidden_layers: int = 2,
        activation=torch.nn.ReLU,
    ):
        super(LinearNet, self).__init__()
        self._hidden_dim = H
        self._hidden_layers = hidden_layers
        self._activation = activation
        self._out_dim = D_out

        self.layers = []
        dim = D_in
        for _ in range(self._hidden_layers):
            self.layers.append(torch.nn.Linear(dim, self._hidden_dim))
            self.layers.append(self._activation())
            dim = self._hidden_dim
        self.layers.append(torch.nn.Linear(dim, self._out_dim))
        self.model = torch.nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor):
        x = x.requires_grad_(True)
        return self.model(x)


class NNTrainer(Trainer):
    def __init__(self, device=None):
        super().__init__()
        self._device = device
        self._loss_fn: Optional[torch.nn.MSELoss] = None

    @property
    def name(self) -> str:
        return "linear_net"

    def train(
        self,
        data: TrainingData,
        iterations: int = 100,
        epochs: int = 1,
        num_samples: int = 0,
    ):
        d_in, d_out = (
            data.train_x.shape[1],
            data.train_y.shape[1] if len(data.train_y.shape) > 1 else 1,
        )
        if d_in == 0 or d_out == 0:
            return None
        h = 500
        n = data.train_x.shape[0] // 200

        logging.info(f"start training...")
        logging.info(f"  d_in = {d_in}, h = {h}, d_out = {d_out}, n = {n}")
        st = time.process_time()

        self._model = LinearNet(d_in, h, d_out)
        if self._device is not None and self._device.type == "cuda":
            self._model = self._model.cuda()
        self._loss_fn = torch.nn.MSELoss(reduction="mean")
        learning_rate = 1e-3
        optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=5, verbose=True, threshold=1e-5
        )
        for _ in range(epochs):
            for t in range(iterations):
                x, y, _ = super()._sample(
                    data.train_x, data.train_y, data.train_weight, num_samples, True
                )
                x = torch.as_tensor(x, device=self._device)
                y = torch.as_tensor(y, device=self._device)
                if len(y.shape) == 1:
                    y = y.reshape(-1, 1)
                y_pred = self._model(x)
                # pyre-fixme[29]: `Optional[torch.nn.MSELoss]` is not a function.
                loss = self._loss_fn(y_pred, y)
                if (t + 1) % 10 == 0:
                    scheduler.step(loss.item())
                    logging.info(f"  step [{t + 1}]: loss={loss.item()}")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        logging.info(f"  training time {time.process_time() - st}")

    def predict(self, x: Tensor, device=None) -> PredictResults:
        if self._model is not None:
            self._model.eval()
            proba = torch.as_tensor(self._model(x), dtype=torch.float, device=device)
            return PredictResults(torch.argmax(proba, 1), proba)
        else:
            raise Exception("mode not trained")

    def score(self, x: Tensor, y: Tensor, weight: Optional[Tensor] = None) -> float:
        if self._loss_fn is not None:
            # pyre-fixme[29]: `Optional[torch.nn.MSELoss]` is not a function.
            return self._loss_fn(y, x).item()
        else:
            raise Exception("mode not trained")

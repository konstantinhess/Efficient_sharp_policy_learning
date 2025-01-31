from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
import lightning as L
from src.nuisance_models import PropensityModel, OutcomeModel, ConditionalQuantileModel

class BaselineMinimaxEstimator(L.LightningModule):
    # Baseline minimax estimator via ternary search
    def __init__(self,
                 propensity_model: PropensityModel,
                 gamma: float,
                 baseline_policy: nn.Module = None,
                 lr: float = 1e-3
                 ):

        super(BaselineMinimaxEstimator, self).__init__()
        self.propensity_model = propensity_model

        self.policy = nn.Sequential(
            nn.Linear(propensity_model.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, propensity_model.num_treatments),
        )


        self.baseline_policy = baseline_policy
        self.gamma = gamma
        self.num_treatments = propensity_model.num_treatments
        self.lr = lr

    def forward(self, X, A, Y, true_prop):

        propensity = torch.softmax(self.propensity_model(X), dim=1)

        policy = torch.softmax(self.policy(X), dim=1)
        # Randomized baseline policy
        if not self.baseline_policy:
            baseline = torch.ones(policy.shape) / policy.shape[1]
        else:
            baseline = torch.softmax(self.baseline_policy(X), dim=1)

        # Lower and upper bound on inverse propensity score
        lower_bound = 1 + (self.gamma**(-1)) * ((propensity[A==1]**(-1)) -1)
        upper_bound = 1 + (self.gamma) * ((propensity[A==1]**(-1)) - 1)

        # Reward
        R = (policy - baseline)[A == 1] * Y.flatten()

        # Perform ternary search
        ordering = torch.argsort(R)

        lambda_k = (upper_bound[ordering] * R[ordering]).sum() / upper_bound[ordering].sum()

        k = 0
        k_star = 0
        while k < len(R):
            numerator = (lower_bound[ordering[:k+1]] * R[ordering[:k+1]]).sum() + (upper_bound[ordering[k+1:]] * R[ordering[k+1:]]).sum()
            denominator = (lower_bound[ordering[:k+1]]).sum() + (upper_bound[ordering[k+1:]]).sum()
            lambda_next =  numerator / denominator

            if lambda_next < lambda_k:
                k_star = k
                break

            lambda_k = lambda_next
            k += 1

        if k == len(R):
            k_star = len(R) - 1

        W = torch.concat((lower_bound[ordering[:k_star+1]], upper_bound[ordering[k_star+1:]]), dim=-1)

        return W, ordering

    def training_step(self, batch, batch_idx):
        X, A, Y, true_prop = batch
        W, ordering = self(X, A, Y, true_prop)
        # Subgradient

        G = (W * (Y.flatten()[ordering]) * ((torch.softmax(self.policy(X), dim=-1)[A==1])[ordering])/ torch.sum(W)).sum(dim=0)

        self.log("train_G", G, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return G

    def validation_step(self, batch, batch_idx):
        X, A, Y, true_prop = batch
        W, ordering = self(X, A, Y, true_prop)
        # Subgradient

        G = (W * (Y.flatten()[ordering]) * ((torch.softmax(self.policy(X), dim=-1)[A==1])[ordering])/ torch.sum(W)).sum(dim=0)

        self.log("val_G", G, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return G

    def test_step(self, batch, batch_idx):
        X, A, Y, _ = batch
        W, ordering = self(X, A, Y)
        baseline = torch.ones(A.shape) / A.shape[1]

        # Subgradient
        G = (W * ((torch.softmax(self.policy(X), dim=-1)[A==1] - baseline[A==1])[ordering]) / torch.sum(W)).sum(dim=0)
        self.log("G", G)

    def predict_step(self, batch, batch_idx):
        X, _, _, _, = batch
        return torch.softmax(self.policy(X), dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        return optimizer

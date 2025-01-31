from typing import Any

import torch
from torch import nn
import lightning as L
from src.nuisance_models import PropensityModel, OutcomeModel

class ValuePlugInEstimator(L.LightningModule):
    # Plug-in estimator of bound of value function
    def __init__(self,
                 propensity_model: PropensityModel,
                 outcome_model_lower: OutcomeModel, # \underbar
                 outcome_model_upper: OutcomeModel, # \bar
                 gamma: float,
                 bound_type: str = 'upper',  # upper / lower
                 lr: float = 1e-3
                 ):

        super(ValuePlugInEstimator, self).__init__()
        self.propensity_model = propensity_model
        self.outcome_model_lower = outcome_model_lower
        self.outcome_model_upper = outcome_model_upper

        self.policy = nn.Sequential(
            nn.Linear(propensity_model.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, propensity_model.num_treatments),
        )

        self.gamma = gamma
        self.num_treatments = propensity_model.num_treatments
        self.lr = lr

        if bound_type == 'upper':
            self.sign = 1
            assert self.outcome_model_upper.quantile == gamma / (1. + gamma)
            assert self.outcome_model_lower.quantile == gamma / (1. + gamma)

        elif bound_type == 'lower':
            self.sign = -1
            assert self.outcome_model_upper.quantile == 1. / (1. + gamma)
            assert self.outcome_model_lower.quantile == 1. / (1. + gamma)

        else:
            raise ValueError("bound_type must be either 'upper' or 'lower'")

    def forward(self, X, A):

        b_plus = 1 - self.gamma**self.sign
        b_minus = 1 - self.gamma ** (-self.sign)

        propensity = torch.softmax(self.propensity_model(X), dim=1)
        policy = torch.softmax(self.policy(X), dim=1)

        Q_pi = torch.zeros(size=(X.shape[0], 1))

        for i in range(0, self.num_treatments):

            # Create one-hot encoded treatment vector
            a = torch.zeros(size=A.shape)
            a[:, i] = 1

            # 1) Compute sharp bound of CAPO

            c_plus = b_plus * propensity[:, i:(i+1)] + self.gamma ** self.sign
            c_minus = b_minus * propensity[:, i:(i+1)] + self.gamma ** (-self.sign)

            mu_underbar = self.outcome_model_lower(X, a)
            mu_bar = self.outcome_model_upper(X, a)

            # Sharp CAPO bound
            Q = c_minus * mu_underbar + c_plus * mu_bar

            # 2) Compute sum over all treatments of Q(a,x)*\pi(a|x)
            Q_pi += Q * policy[:, i:(i+1)]

        # 3) Compute the value bound by averaging over samples
        V_bound = Q_pi.mean(dim=0)

        return V_bound

    def training_step(self, batch, batch_idx):
        X, A, _, _ = batch

        V_bound = self(X, A)
        self.log("train_V_bound", V_bound, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return V_bound

    def validation_step(self, batch, batch_idx):
        X, A, _, _ = batch

        V_bound = self(X, A)

        self.log("val_V_bound", V_bound, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    def test_step(self, batch, batch_idx):
        X, A, _, _ = batch

        V_bound = self(X, A)
        self.log("test_V_bound", V_bound)

    def predict_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        return optimizer


from typing import Any

import torch
from torch import nn
import lightning as L
from src.nuisance_models import PropensityModel, OutcomeModel

class ValueEfficientEstimator(L.LightningModule):
    # Efficient estimator of bound on value function
    def __init__(self,
                 propensity_model: PropensityModel,
                 outcome_model_lower: OutcomeModel, # \underbar
                 outcome_model_upper: OutcomeModel, # \bar
                 gamma: float,
                 bound_type: str = 'upper',  # upper / lower
                 lr: float = 1e-3
                 ):

        super(ValueEfficientEstimator, self).__init__()
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

        self.alpha = gamma / (1+gamma)
        self.gamma = gamma
        self.num_treatments = propensity_model.num_treatments
        self.lr = lr

        if bound_type == 'upper':
            self.sign = 1
        elif bound_type == 'lower':
            self.sign = -1
        else:
            raise ValueError("bound_type must be either 'upper' or 'lower'")

    def forward(self, X, A, Y):

        b_plus = 1 - self.gamma**self.sign
        b_minus = 1 - self.gamma ** (-self.sign)

        propensity = torch.softmax(self.propensity_model(X), dim=1)
        policy = torch.softmax(self.policy(X), dim=1)



        # 1) First term in the equation:
        # \sum_a \pi(a|X) [Q(a,X)-e(a,X)(b^{-}\underbar{\mu}^{+}(a,X)+b^{+}\bar{\mu}^{+}(a,X))]

        first_term = torch.zeros(size=(X.shape[0], 1))
        for i in range(0, self.num_treatments):

            # Create one-hot encoded treatment vector
            a = torch.zeros(size=A.shape)
            a[:, i] = 1

            # a) Compute sharp bound of CAPO

            c_plus = b_plus * propensity[:, i:(i+1)] + self.gamma ** self.sign
            c_minus = b_minus * propensity[:, i:(i+1)] + self.gamma ** (-self.sign)

            mu_underbar = self.outcome_model_lower(X, a)
            mu_bar = self.outcome_model_upper(X, a)

            # Sharp CAPO bound
            Q = c_minus * mu_underbar + c_plus * mu_bar

            # b) Compute policy-weighted sum over all treatments
            first_term += policy[:, i:(i+1)] *\
                    (Q - propensity[:, i:(i+1)] * (b_minus * mu_underbar + b_plus * mu_bar))

        # 2) Second term in the equation:
        # \pi(A|X)*(b^{-}\underbar{\mu}^{+}(A,X)+b^{+}\bar{\mu}^{+}(A,X))

        second_term = policy[A==1].unsqueeze(dim=-1) * (b_minus * self.outcome_model_lower(X, A) + b_plus * self.outcome_model_upper(X, A))


        # 3) Third term in the equation:
        # \pi(A|X)/e(A,X) (c^{-}(A,X)-c^{+}(A,X))(F_{X,A}^{-1}(\alpha^{+})(\underbar{\Delta}_{Y,X,A}^{+}-\alpha^{+}))

        Delta_ubar = (Y <= self.outcome_model_lower.quantile_model(X, A)) * 1
        Delta_bar = (Y >= self.outcome_model_upper.quantile_model(X, A)) * 1
        F_XA_inv_plus = self.outcome_model_lower.quantile_model(X, A)
        c_minus = b_minus * propensity[A==1].unsqueeze(dim=-1) + self.gamma ** (-self.sign)
        c_plus = b_plus * propensity[A==1].unsqueeze(dim=-1) + self.gamma ** (self.sign)

        third_term = policy[A==1].unsqueeze(dim=-1) / propensity[A==1].unsqueeze(dim=-1) *\
                        (c_minus - c_plus) * F_XA_inv_plus * (Delta_ubar - self.alpha)


        # 4) Fourth term in the equation:
        # \pi(A|X)/e(A,X) (c^{-}(A,X)(Y\underbar{\Delta}^{+}_{Y,X,A}-\underbar{\mu}^{+}(A,X)) + c^{+}(A,X)(Y\bar{\Delta}^{+}_{Y,X,A}-\bar{\mu}^{+}(A,X)))

        mu_ubar = self.outcome_model_lower(X, A)
        mu_bar = self.outcome_model_upper(X, A)
        fourth_term = policy[A==1].unsqueeze(dim=-1) / propensity[A==1].unsqueeze(dim=-1) *\
                        (c_minus * (Y * Delta_ubar - mu_ubar) + c_plus * (Y * Delta_bar - mu_bar))


        # 5) Compute the value bound by averaging over samples
        V_bound = (first_term + second_term + third_term + fourth_term).mean(dim=0)

        return V_bound

    def training_step(self, batch, batch_idx):
        X, A, Y, _ = batch

        V_bound = self(X, A, Y)
        self.log("train_V_bound", V_bound)
        return V_bound

    def validation_step(self, batch, batch_idx):
        X, A, Y, _ = batch

        V_bound = self(X, A, Y)
        self.log("val_V_bound", V_bound)

    def test_step(self, batch, batch_idx):
        X, A, Y, _ = batch

        V_bound = self(X, A, Y)
        self.log("test_V_bound", V_bound)

    def predict_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        return optimizer


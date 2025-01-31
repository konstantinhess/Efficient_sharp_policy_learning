import torch
from torch import nn
import lightning as L
from src.nuisance_models import PropensityModel

class IPWEstimator(L.LightningModule):
    # IPW estimator
    def __init__(self,
                 propensity_model: PropensityModel,
                 lr: float = 1e-3
                 ):

        super(IPWEstimator, self).__init__()
        self.propensity_model = propensity_model

        self.policy = nn.Sequential(
            nn.Linear(propensity_model.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, propensity_model.num_treatments),
        )

        self.num_treatments = propensity_model.num_treatments
        self.lr = lr
    def forward(self, X, A, Y):

        propensity = torch.softmax(self.propensity_model(X), dim=1)
        policy = torch.softmax(self.policy(X), dim=1)

        V = (policy[A == 1].unsqueeze(dim=-1) * Y / propensity[A == 1].unsqueeze(dim=-1)).mean(dim=0)

        return V

    def training_step(self, batch, batch_idx):
        X, A, Y, _ = batch
        V = self(X, A, Y)

        self.log("train_V", V, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return V

    def validation_step(self, batch, batch_idx):
        X, A, Y, _ = batch

        V = self(X, A, Y)
        self.log("val_V", V, on_step=False, on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx):
        X, A, Y, _ = batch

        V = self(X, A, Y)
        self.log("test_V", V)

    def predict_step(self, batch, batch_idx):
        X, _, _, _, = batch
        return torch.softmax(self.policy(X), dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        return optimizer


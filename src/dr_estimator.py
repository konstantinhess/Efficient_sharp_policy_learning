import torch
from torch import nn
import lightning as L
from src.nuisance_models import OutcomeModel
from src.nuisance_models import PropensityModel
import mlflow
class DREstimator(L.LightningModule):
    # Direct method
    def __init__(self,
                 propensity_model: PropensityModel,
                 outcome_model: OutcomeModel,
                 lr: float = 1e-3
                 ):

        super(DREstimator, self).__init__()
        self.outcome_model = outcome_model
        self.propensity_model = propensity_model
        assert outcome_model.type == 'standard', 'DirectMethod only supports standard outcome models'


        self.policy = nn.Sequential(
            nn.Linear(outcome_model.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, outcome_model.num_treatments),
        )

        self.num_treatments = outcome_model.num_treatments
        self.lr = lr

    def forward(self, X, A, Y):

        propensity = torch.softmax(self.propensity_model(X), dim=1)
        policy = torch.softmax(self.policy(X), dim=1)

        V = torch.zeros(size=(X.shape[0], 1))
        for i in range(0, self.num_treatments):

            # Create one-hot encoded treatment vector
            a = torch.zeros(size=A.shape)
            a[:, i] = 1

            indicator = a[:, i:i+1] == A[:, i:i+1]

            # Compute CAPO and estimate the value function
            capo = self.outcome_model(X, a)

            V += policy[:, i:(i+1)] * (capo + indicator / propensity[:, i:(i+1)] * (Y - capo))

        V = V.mean(dim=0)

        return V

    def training_step(self, batch, batch_idx):
        X, A, Y, _ = batch
        V = self(X, A, Y)

        # Log the training loss to MLflow
        #mlflow.log_metric("train_loss", V.item(), step=self.global_step)

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


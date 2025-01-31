import torch
from torch import nn
import lightning as L


# 1) Propensity score estimator

class PropensityModel(L.LightningModule):
    def __init__(self, input_dim: int, num_treatments: int, lr: float = 1e-3):
        super(PropensityModel, self).__init__()

        self.input_dim = input_dim
        self.num_treatments = num_treatments

        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_treatments)  # Output size matches number of treatment options
        )
        self.criterion = nn.CrossEntropyLoss()  # Cross-Entropy Loss for multi-class classification
        self.lr = lr
    def forward(self, X):
        return self.model(X)

    def training_step(self, batch, batch_idx):
        X, A, _, _ = batch
        A = torch.argmax(A, dim=1)  # Convert one-hot encoded A to class indices
        A_pred = self(X)  # Raw logits
        loss = self.criterion(A_pred, A)
        self.log("train_prop_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, A, _, _ = batch
        A = torch.argmax(A, dim=1)  # Convert one-hot encoded A to class indices
        A_pred = self(X)  # Raw logits
        loss = self.criterion(A_pred, A)
        self.log("val_prop_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    def test_step(self, batch, batch_idx):
        X, A, _, _ = batch
        A = torch.argmax(A, dim=1)  # Convert one-hot encoded A to class indices
        A_pred = self(X)  # Raw logits
        loss = self.criterion(A_pred, A)
        self.log("test_loss", loss)

    def predict_step(self, batch, batch_idx):
        X, _, _, true_propensity = batch
        return torch.softmax(self(X), dim=1), true_propensity  # Convert logits to probabilities

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

########################################################################################################################

# 2) Conditional quantile model

class QuantileLoss(nn.Module):
    def __init__(self, quantile: float):
        super(QuantileLoss, self).__init__()
        self.quantile = quantile

    def forward(self, y_pred, y_true):
        error = y_true - y_pred
        loss = torch.max((self.quantile - 1) * error, self.quantile * error)
        return loss.mean()

class ConditionalQuantileModel(L.LightningModule):
    def __init__(self, quantile: float, input_dim: int, n_treatments: int, lr: float = 1e-3):
        super(ConditionalQuantileModel, self).__init__()
        self.quantile = quantile

        self.model = nn.Sequential(
            nn.Linear(input_dim + n_treatments, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.criterion = QuantileLoss(quantile)
        self.lr = lr
    def forward(self, X, A):
        inputs = torch.cat([X, A], dim=1)
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        X, A, Y, _ = batch
        Y_pred = self(X, A)
        loss = self.criterion(Y_pred, Y)
        self.log("train_quantile_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, A, Y, _ = batch
        Y_pred = self(X, A)
        loss = self.criterion(Y_pred, Y)
        self.log("val_quantile_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        X, A, Y, _ = batch
        Y_pred = self(X, A)
        loss = self.criterion(Y_pred, Y)
        self.log("test_loss", loss)

    def predict_step(self, batch, batch_idx):
        X, A, Y, _ = batch
        return self(X, A)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)


########################################################################################################################

# 3) Outcome model

class OutcomeModel(L.LightningModule):
    def __init__(self, input_dim: int, num_treatments: int, type: str, quantile: float, quantile_model: ConditionalQuantileModel = None, lr: float = 1e-3):
        super(OutcomeModel, self).__init__()

        # need quantile model for masked regression
        self.quantile_model = quantile_model
        self.input_dim = input_dim
        self.num_treatments = num_treatments

        assert type in ['upper', 'lower', 'standard'], "Type must be in ['upper', 'lower', 'standard']" # upper = "\bar", lower = "\underbar"
        self.type = type

        if type in ['upper', 'lower']:
            assert quantile == self.quantile_model.quantile
        self.quantile = quantile

        self.model = nn.Sequential(
            nn.Linear(input_dim+num_treatments, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.criterion = nn.MSELoss()
        self.lr = lr
    def forward(self, X, A):
        inputs = torch.cat([X, A], dim=1)
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        X, A, Y, _ = batch

        if self.type == 'upper':
            quantile_pred = self.quantile_model(X, A)
            mask = (Y >= quantile_pred) * 1
            Y_masked = Y * mask
        elif self.type == 'lower':
            quantile_pred = self.quantile_model(X, A)
            mask = (Y <= quantile_pred) * 1
            Y_masked = Y * mask
        else: # for direct method
            Y_masked = Y

        Y_pred = self(X, A)
        loss = self.criterion(Y_pred, Y_masked)
        self.log("train_"+self.type+"_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, A, Y, _ = batch

        if self.type == 'upper':
            quantile_pred = self.quantile_model(X, A)
            mask = (Y >= quantile_pred) * 1
            Y_masked = Y * mask
        elif self.type == 'lower':
            quantile_pred = self.quantile_model(X, A)
            mask = (Y <= quantile_pred) * 1
            Y_masked = Y * mask
        else:  # for direct method
            Y_masked = Y

        Y_pred = self(X, A)
        loss = self.criterion(Y_pred, Y_masked)
        self.log("val_" + self.type + "_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    def test_step(self, batch, batch_idx):
        X, A, Y, _ = batch

        if self.type == 'upper':
            quantile_pred = self.quantile_model(X, A)
            mask = (Y >= quantile_pred) * 1
            Y_masked = Y * mask
        elif self.type == 'lower':
            quantile_pred = self.quantile_model(X, A)
            mask = (Y <= quantile_pred) * 1
            Y_masked = Y * mask
        else:  # for direct method
            Y_masked = Y

        Y_pred = self(X, A)
        loss = self.criterion(Y_pred, Y_masked)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)


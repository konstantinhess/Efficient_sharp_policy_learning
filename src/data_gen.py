import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch

class SyntheticDataSimulator:
    def __init__(self, n_samples, n_features, n_treatments, gamma, seed):
        """
        Initialize the synthetic data simulator.

        Args:
            n_samples (int): Number of samples to generate.
            n_features (int): Number of features to generate.
            n_treatments (int): Number of treatments (>=2).
            conf_std (float): Standard deviation for confounding variable.
            seed (int): Random seed for reproducibility.
        """
        assert n_treatments > 1, "Number of treatments must be greater than 1 (binary treatment corresponds to n_treatments=2)."
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_treatments = n_treatments
        self.gamma = gamma
        self.seed = seed
        self.train=True

    def generate(self) -> pd.DataFrame:
        """
        Generate synthetic data for causal inference.

        Returns:
            pd.DataFrame: Dataframe containing the generated data.
        """
        np.random.seed(self.seed)


        epsilon = np.random.randn(self.n_samples)

        X = self._generate_covariates()
        U = self._generate_unobserved_confounders()
        true_propensity = self._generate_true_propensity(X, U)
        A = self._generate_treatment_one_hot(true_propensity)
        Y = self._generate_outcomes(X, A, U, epsilon)

        # Combine into a dataframe
        data = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(self.n_features)])
        for i in range(A.shape[1]):
            data[f"A{i}"] = A[:, i]
        data['Y'] = Y
        data['true_propensity'] = true_propensity

        self.X = X
        self.U = U
        self.epsilon = epsilon
        return data


    def _generate_covariates(self) -> np.ndarray:
        """
        Generate covariates (X) for the dataset.

        Returns:
            np.ndarray: Generated covariates of shape (n_samples, n_features).
        """
        X = np.random.uniform(low=-2, high=2, size=(self.n_samples, self.n_features))
        return X

    def _generate_unobserved_confounders(self) -> np.ndarray:
        """
        Generate unobserved confounders (U).

        Returns:
            np.ndarray: Generated unobserved confounders of shape (n_samples,).
        """
        U = np.random.binomial(1, 0.5, (self.n_samples, ))
        return U

    def _generate_outcomes(self, X: np.ndarray, A: np.ndarray, U, epsilon) -> np.ndarray:
        """
        Generate outcomes (Y) based on X, U, and A.

        Args:
            X (np.ndarray): Covariates.
            U (np.ndarray): Unobserved confounders.
            A (np.ndarray): Treatment assignments.

        Returns:
            np.ndarray: Generated outcomes of shape (n_samples,).
        """
        Y =  (2*np.argmax(A, axis=-1) -1) * (X.mean(axis=-1) +1) - 2*np.sin(2*(2*np.argmax(A, axis=-1)-1)*X.mean(axis=-1)) - 2*(2*U-1)*(1+0.5*X.mean(axis=-1))+epsilon
        if self.train:
            Y = (Y-Y.mean())/Y.std()
        return Y



    def _generate_true_propensity(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        """
        Generate true propensity scores based on X and U.

        Args:
            X (np.ndarray): Covariates.
            U (np.ndarray): Unobserved confounders.

        Returns:
            np.ndarray: Propensity scores of shape (n_samples,).
        """
        nominal_propensity = 1/(1+np.exp(-0.75*X.mean(axis=-1)+0.5))

        lb = 1 + (self.gamma**(-1)) * (1/nominal_propensity -1)
        ub = 1 + self.gamma * (1/nominal_propensity -1)

        true_propensity = U / lb + (1-U) / ub

        return true_propensity

    def _generate_treatment_one_hot(self, propensity) -> np.ndarray:
        """
        Generate a one-hot encoded array for treatments.

        Returns:
            np.ndarray: One-hot encoded treatment array of shape (n_samples, n_treatments).
        """
        treatment = np.random.binomial(1, propensity)

        one_hot_array = np.zeros((self.n_samples, self.n_treatments), dtype=int)
        for i in range(self.n_samples):
            k = treatment[i]
            one_hot_array[i, k] = 1

        return one_hot_array


    def evaluate_policy(self, policy_probs) -> pd.DataFrame:
        """
        Evaluate policy.

        Args:
            policy_probs (np.ndarray): Policy probabilities.
            X (np.ndarray): Covariates.

        Returns:
            (np.ndarray): Value of policy
        """
        np.random.seed(self.seed)
        self.train = False

        pi_Y = np.zeros((policy_probs.shape[0], ))
        for i in range(policy_probs.shape[1]):
            a = np.zeros(policy_probs.shape)
            a[:, i] = 1
            # Expected potential outcome
            Y = self._generate_outcomes(self.X, a,self.U, 0)
            # Weight by policy probability
            pi_Y += policy_probs[:, i] * Y

        # Value of policy
        V = pi_Y.mean(axis=0)
        return V




########################################################################################################################


# Torch Dataset class
class CustomDataset(Dataset):
    def __init__(self, data):

        self.X = torch.tensor(data[[col for col in data.columns if col.startswith("X")]].values, dtype=torch.float32)
        self.A = torch.tensor(data[[col for col in data.columns if col.startswith("A")]].values, dtype=torch.float32)
        self.Y = torch.tensor(data['Y'].values, dtype=torch.float32).unsqueeze(1)
        # not used
        self.true_propensity = torch.tensor(data['true_propensity'].values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.A[idx], self.Y[idx], self.true_propensity[idx]


def create_dataloader(dataset, batch_size=32, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

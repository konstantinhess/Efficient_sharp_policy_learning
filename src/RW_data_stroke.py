import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class RWData:
    def __init__(self, file_path, seed, n_samples_train_policy, n_samples_train_nuisance, n_samples_test, selection_bias):
        """
        Initialize the real-world data handler.

        Args:
            file_path (str): Path to the pre-processed CSV file.
            seed (int): Random seed for reproducibility.
            n_samples_train_policy (int): Number of training samples for policy learning.
            n_samples_train_nuisance (int): Number of training samples for nuisance estimation.
            n_samples_test (int): Number of test samples.
        """
        self.file_path = file_path
        self.seed = seed
        self.n_samples_train_policy = n_samples_train_policy
        self.n_samples_train_nuisance = n_samples_train_nuisance
        self.n_samples_test = n_samples_test
        self.selection_bias = selection_bias
        self._load_data()


    def _apply_selection_bias(self, df, confounder_col):
        """Apply selection bias based on individual-level confounder values for multiple treatment groups."""
        np.random.seed(self.seed)
        mean_conf = df[confounder_col].mean()

        # Mask for TREATMENT_0
        mask_treat_0 = (df["TREATMENT_0"] == 1) & (df[confounder_col] > mean_conf)
        drop_indices_0 = df[mask_treat_0].sample(frac=self.selection_bias, random_state=self.seed).index

        # Mask for TREATMENT_2
        mask_treat_2 = (df["TREATMENT_1"] == 1) & (df[confounder_col] < mean_conf)
        drop_indices_2 = df[mask_treat_2].sample(frac=self.selection_bias, random_state=self.seed).index

        # Drop selected indices
        df = df.drop(index=drop_indices_0.union(drop_indices_2))

        # Drop confounder column
        df = df.drop(columns=[confounder_col])

        return df

    def _load_data(self):
        """Load and preprocess the dataset."""
        np.random.seed(self.seed)
        df = pd.read_csv(self.file_path)
        df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)  # Shuffle dataset

        # Convert categorical variables to numeric values
        categorical_cols = ["SEX", "RCONSC", "RATRIAL", "RDEF4", "STYPE"]
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        df_train = df[:-self.n_samples_test].copy() # self.n_samples_train_policy+self.n_samples_train_nuisance].copy()
        df_test = df.iloc[-self.n_samples_test:].copy()


        # Introduce hidden confounding for training sets
        confounder_col = "RSBP"
        df_train = self._apply_selection_bias(df_train, confounder_col)

        # Split into three disjoint sets
        df_train_policy = df_train.iloc[:self.n_samples_train_policy].copy()
        df_train_nuisance = df_train.iloc[
                            self.n_samples_train_policy:self.n_samples_train_policy + self.n_samples_train_nuisance].copy()


        # Remove the confounder from the test set but do not introduce bias
        df_test.drop(columns=[confounder_col], inplace=True)

        # Convert everything to float
        df_train_policy = df_train_policy.astype(float)
        df_train_nuisance = df_train_nuisance.astype(float)
        df_test = df_test.astype(float)

        # Extract numpy arrays for each set
        X_train_policy = df_train_policy.drop(
            columns=["TD"] + [col for col in df.columns if col.startswith("TREATMENT_")]).values
        X_mean, X_std = X_train_policy.mean(axis=0), X_train_policy.std(axis=0)
        self.X_train_policy = (X_train_policy - X_mean) / X_std
        self.A_train_policy = df_train_policy[
            [col for col in df_train_policy.columns if col.startswith("TREATMENT_")]].values
        Y_train_policy = -df_train_policy["TD"].values # swap sign since we minimize
        Y_mean, Y_std = Y_train_policy.mean(), Y_train_policy.std()
        self.Y_train_policy = (Y_train_policy - Y_mean) / Y_std

        X_train_nuisance = df_train_nuisance.drop(
            columns=["TD"] + [col for col in df.columns if col.startswith("TREATMENT_")]).values
        self.X_train_nuisance = (X_train_nuisance - X_mean) / X_std
        self.A_train_nuisance = df_train_nuisance[
            [col for col in df_train_nuisance.columns if col.startswith("TREATMENT_")]].values
        Y_train_nuisance = -df_train_nuisance["TD"].values # swap sign since we minimize
        self.Y_train_nuisance = (Y_train_nuisance - Y_mean) / Y_std

        X_test = df_test.drop(
            columns=["TD"] + [col for col in df.columns if col.startswith("TREATMENT_")]).values
        self.X_test = (X_test - X_mean) / X_std
        self.A_test = df_test[[col for col in df_test.columns if col.startswith("TREATMENT_")]].values
        Y_test = df_test["TD"].values
        self.Y_test = Y_test # do not normalize outcomes for testing --> want ground truth time to death


    def _generate_ipw_outcomes(self, A):
        """
        Inverse propensity weighted outcome (propensity scores known from RCT study)."""

        if A.argmax() == 0:
            rct_propensity_score = 1/2
        elif A.argmax() == 1:
            rct_propensity_score = 1/4
        elif A.argmax() == 2:
            rct_propensity_score = 1/8
        else:
            rct_propensity_score = 1/8

        return self.Y_test / rct_propensity_score

    def evaluate_policy(self, policy_probs):
        """
        Evaluate a policy given probability distributions over treatments.

        Args:
            policy_probs (np.ndarray): Policy probabilities.

        Returns:
            float: Value of the policy (expected outcome).
        """
        np.random.seed(self.seed)

        pi_Y = np.zeros(policy_probs.shape[0])
        for i in range(policy_probs.shape[1]):
            a = np.zeros(policy_probs.shape)
            a[:, i] = 1
            Y = self._generate_ipw_outcomes(a)
            pi_Y += policy_probs[:, i] * Y

        return pi_Y.mean()



# Torch Dataset class
class CustomDataset(Dataset):
    def __init__(self, X, A, Y):

        self.X = torch.tensor(X, dtype=torch.float32)
        self.A = torch.tensor(A, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)

        true_propensity = torch.zeros(A.shape)
        true_propensity[:, 0] = 1/2
        true_propensity[:, 1] = 1/4
        true_propensity[:, 2] = 1/8
        true_propensity[:, 3] = 1/8
        self.true_propensity = true_propensity

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.A[idx], self.Y[idx], self.true_propensity[idx]


def create_dataloader(dataset, batch_size=32, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

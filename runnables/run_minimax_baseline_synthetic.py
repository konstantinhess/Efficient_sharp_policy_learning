import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import numpy as np
import torch
import mlflow
import mlflow.pytorch
from src.nuisance_models import PropensityModel, OutcomeModel
from src.baseline_minimax_estimator import BaselineMinimaxEstimator
from src.data_gen import SyntheticDataSimulator, CustomDataset, create_dataloader
from torch.utils.data import random_split

# Enable the `eval` resolver
OmegaConf.register_new_resolver("eval", eval)

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # Set seed
    L.fabric.seed_everything(cfg.exp.global_seed)

    ####################################################################################################################
    # MLflow Setup
    if cfg.exp.logging:
        mlflow.set_tracking_uri(cfg.exp.mlflow_uri)
        mlflow.set_experiment("Synthetic_Baseline_Minimax")
        # mlflow_logger = L.pytorch.loggers.MLFlowLogger(experiment_name="Synthetic_Baseline_Minimax",
        #                                                 tracking_uri=cfg.exp.mlflow_uri)
        with mlflow.start_run():
            # Log parameters under `exp`
            mlflow.log_params(OmegaConf.to_container(cfg.exp, resolve=True))

            # Log parameters under `data`
            mlflow.log_params(OmegaConf.to_container(cfg.data.synthetic, resolve=True))

            # Log parameters
            mlflow.log_params(OmegaConf.to_container(cfg.models.policy.baseline_minimax, resolve=True))


            ############################################################################################################
            # 1) Data
            n_samples_nuisance = int(cfg.data.synthetic.n_samples_train * cfg.data.synthetic.nuisance_split)
            n_samples_policy = cfg.data.synthetic.n_samples_train - n_samples_nuisance

            # Train + val data for nuisance
            generator_nuisance = SyntheticDataSimulator(n_samples_nuisance, cfg.data.synthetic.n_features, cfg.data.synthetic.n_treatments, cfg.data.synthetic.gamma_data, cfg.exp.global_seed)
            data_nuisance = generator_nuisance.generate()
            # Train + val data for policy models
            generator_policy = SyntheticDataSimulator(n_samples_policy, cfg.data.synthetic.n_features, cfg.data.synthetic.n_treatments, cfg.data.synthetic.gamma_data, cfg.exp.global_seed)
            data_policy = generator_policy.generate()
            # Test data for policy models
            generator_test = SyntheticDataSimulator(cfg.data.synthetic.n_samples_test, cfg.data.synthetic.n_features, cfg.data.synthetic.n_treatments, cfg.data.synthetic.gamma_data, cfg.exp.global_seed)
            data_test = generator_test.generate()

            # Create datasets
            dataset_nuisance = CustomDataset(data_nuisance)
            nuisance_train, nuisance_val, _ = random_split(dataset_nuisance,
                                                           [int((1 - cfg.data.synthetic.val_split) * n_samples_nuisance),
                                                            int(cfg.data.synthetic.val_split * n_samples_nuisance), 0])
            nuisance_train_loader = create_dataloader(nuisance_train, batch_size=cfg.data.synthetic.batch_size)
            nuisance_val_loader = create_dataloader(nuisance_val, batch_size=cfg.data.synthetic.batch_size)

            dataset_policy = CustomDataset(data_policy)
            policy_train, policy_val, _ = random_split(dataset_policy,
                                                           [int((1 - cfg.data.synthetic.val_split) * n_samples_policy),
                                                            int(cfg.data.synthetic.val_split * n_samples_policy), 0])
            dataset_test = CustomDataset(data_test)
            _, _, policy_test = random_split(dataset_test, [0, 0, cfg.data.synthetic.n_samples_test])
            policy_train_loader = create_dataloader(policy_train, batch_size=cfg.data.synthetic.batch_size)#int((1 - cfg.data.synthetic.val_split) * n_samples_policy))
            policy_val_loader = create_dataloader(policy_val, batch_size=cfg.data.synthetic.batch_size)#int(cfg.data.synthetic.val_split * n_samples_policy))#

            ############################################################################################################

            # 2) Nuisance models
            prop_early_stopping = EarlyStopping(monitor="val_prop_loss", patience=cfg.exp.patience, mode="min")

            propensity_model = PropensityModel(input_dim=cfg.models.nuisance.propensity_model.input_dim,
                                               num_treatments=cfg.models.nuisance.propensity_model.n_treatments,
                                               lr=cfg.models.nuisance.propensity_model.lr)

            propensity_trainer = L.Trainer(max_epochs=cfg.models.nuisance.propensity_model.num_epochs,
                                           callbacks=[prop_early_stopping])
            propensity_trainer.fit(propensity_model, nuisance_train_loader, nuisance_val_loader)


            ############################################################################################################

            # 3) Policy model

            baseline_minimax_early_stopping = EarlyStopping(monitor="val_G", patience=cfg.exp.patience, mode="min")

            baseline_minimax_model = BaselineMinimaxEstimator(propensity_model=propensity_model,
                                                              gamma=cfg.models.policy.baseline_minimax.gamma,
                                                              baseline_policy=cfg.models.policy.baseline_minimax.baseline_policy,
                                                              lr=cfg.models.policy.baseline_minimax.lr)

            baseline_minimax_trainer = L.Trainer(max_epochs=cfg.models.policy.baseline_minimax.num_epochs,
                                   callbacks=[baseline_minimax_early_stopping])#, logger=mlflow_logger)
            baseline_minimax_trainer.fit(baseline_minimax_model, policy_train_loader, policy_val_loader)

            ############################################################################################################
            # 4) Evaluate
            baseline_minimax_predictions = torch.softmax(baseline_minimax_model.policy(torch.tensor(generator_test.X).float()), dim=1).detach().numpy()
            baseline_minimax_value = generator_test.evaluate_policy(baseline_minimax_predictions)
            randomized_value = generator_test.evaluate_policy(np.ones(baseline_minimax_predictions.shape) / baseline_minimax_predictions.shape[1])

            regret = baseline_minimax_value - randomized_value

            # Log regret as a metric
            mlflow.log_metric("regret", regret)

if __name__ == "__main__":
    main()

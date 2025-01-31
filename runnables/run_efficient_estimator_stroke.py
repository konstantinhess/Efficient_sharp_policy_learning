import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import numpy as np
import torch
import mlflow
import mlflow.pytorch
from src.nuisance_models import PropensityModel, OutcomeModel, ConditionalQuantileModel
from src.efficient_estimator import ValueEfficientEstimator
from src.RW_data_stroke import RWData, CustomDataset, create_dataloader
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
        mlflow.set_experiment("RWD_Efficient")
        # mlflow_logger = L.pytorch.loggers.MLFlowLogger(experiment_name="RWD_Efficient_test2",
        #                                                tracking_uri=cfg.exp.mlflow_uri)
        with mlflow.start_run():
            # Log parameters under `exp`
            mlflow.log_params(OmegaConf.to_container(cfg.exp, resolve=True))

            # Log parameters under `data`
            mlflow.log_params(OmegaConf.to_container(cfg.data.real_world, resolve=True))

            # Log parameters under `models.policy.sharp_efficient`
            mlflow.log_params(OmegaConf.to_container(cfg.models.policy.sharp_efficient, resolve=True))


            ############################################################################################################
            # 1) Data
            n_samples_nuisance = int(cfg.data.real_world.n_samples_train * cfg.data.real_world.nuisance_split)
            n_samples_policy = cfg.data.real_world.n_samples_train - n_samples_nuisance

            # Train + val data for nuisance
            RWdataset = RWData(cfg.data.real_world.data_path, seed=cfg.exp.global_seed,
                               n_samples_train_policy=n_samples_policy,
                               n_samples_train_nuisance=n_samples_nuisance,
                               n_samples_test=cfg.data.real_world.n_samples_test,
                               selection_bias=cfg.data.real_world.selection_bias)

            # Create datasets
            dataset_nuisance = CustomDataset(RWdataset.X_train_nuisance, RWdataset.A_train_nuisance, RWdataset.Y_train_nuisance)
            nuisance_train, nuisance_val, _ = random_split(dataset_nuisance,
                                                           [int((1 - cfg.data.real_world.val_split) * n_samples_nuisance),
                                                            int(cfg.data.real_world.val_split * n_samples_nuisance), 0])
            nuisance_train_loader = create_dataloader(nuisance_train, batch_size=cfg.data.real_world.batch_size)
            nuisance_val_loader = create_dataloader(nuisance_val, batch_size=cfg.data.real_world.batch_size)

            dataset_policy = CustomDataset(RWdataset.X_train_policy, RWdataset.A_train_policy, RWdataset.Y_train_policy)
            policy_train, policy_val, _ = random_split(dataset_policy,
                                                           [int((1 - cfg.data.real_world.val_split) * n_samples_policy),
                                                            int(cfg.data.real_world.val_split * n_samples_policy), 0])

            dataset_test = CustomDataset(RWdataset.X_test, RWdataset.A_test, RWdataset.Y_test)
            _, _, policy_test = random_split(dataset_test, [0, 0, cfg.data.real_world.n_samples_test])
            policy_train_loader = create_dataloader(policy_train, batch_size=cfg.data.real_world.batch_size)
            policy_val_loader = create_dataloader(policy_val, batch_size=cfg.data.real_world.batch_size)

            ############################################################################################################

            # 2) Nuisance models

            # a) Propensity model
            prop_early_stopping = EarlyStopping(monitor="val_prop_loss", patience=cfg.exp.patience, mode="min")
            propensity_model = PropensityModel(input_dim=cfg.data.real_world.n_features,
                                               num_treatments=cfg.data.real_world.n_treatments,
                                               lr=cfg.models.nuisance.propensity_model.lr)

            propensity_trainer = L.Trainer(max_epochs=cfg.models.nuisance.propensity_model.num_epochs,
                                           callbacks=[prop_early_stopping])#, logger=mlflow_logger)
            propensity_trainer.fit(propensity_model, nuisance_train_loader, nuisance_val_loader)

            # b) Quantile model: alpha^+
            quantile_early_stopping = EarlyStopping(monitor="val_quantile_loss", patience=cfg.exp.patience, mode="min")
            quantile_model_plus = ConditionalQuantileModel(quantile=cfg.models.nuisance.quantile_model.alpha_plus,
                                                           input_dim=cfg.data.real_world.n_features,
                                                           n_treatments=cfg.data.real_world.n_treatments,
                                                           lr=cfg.models.nuisance.quantile_model.lr)

            quantile_plus_trainer = L.Trainer(max_epochs=cfg.models.nuisance.quantile_model.num_epochs,
                                              callbacks=[quantile_early_stopping])#, logger=mlflow_logger)
            quantile_plus_trainer.fit(quantile_model_plus, nuisance_train_loader, nuisance_val_loader)


            # c) Outcome model: \bar{mu}^+
            upper_early_stopping = EarlyStopping(monitor="val_upper_loss", patience=cfg.exp.patience, mode="min")
            outcome_model_upper = OutcomeModel(input_dim=cfg.data.real_world.n_features,
                                         num_treatments=cfg.data.real_world.n_treatments,
                                         type='upper',
                                         quantile=cfg.models.nuisance.quantile_model.alpha_plus,
                                         quantile_model=quantile_model_plus,
                                         lr=cfg.models.nuisance.outcome_model.lr)

            outcome_trainer_upper = L.Trainer(max_epochs=cfg.models.nuisance.outcome_model.num_epochs,
                                        callbacks=[upper_early_stopping])#, logger=mlflow_logger)
            outcome_trainer_upper.fit(outcome_model_upper, nuisance_train_loader, nuisance_val_loader)

            # d) Outcome model: \underbar{mu}+
            lower_early_stopping = EarlyStopping(monitor="val_lower_loss", patience=cfg.exp.patience, mode="min")

            outcome_model_lower = OutcomeModel(input_dim=cfg.data.real_world.n_features,
                                             num_treatments=cfg.data.real_world.n_treatments,
                                             type='lower',
                                             quantile=cfg.models.nuisance.quantile_model.alpha_plus,
                                             quantile_model=quantile_model_plus,
                                             lr=cfg.models.nuisance.outcome_model.lr)

            outcome_trainer_lower = L.Trainer(max_epochs=cfg.models.nuisance.outcome_model.num_epochs,
                                            callbacks=[lower_early_stopping])#, logger=mlflow_logger)
            outcome_trainer_lower.fit(outcome_model_lower, nuisance_train_loader, nuisance_val_loader)

            ############################################################################################################

            # 3) Policy model

            efficient_early_stopping = EarlyStopping(monitor="val_V_bound", patience=cfg.exp.patience, mode="min")

            efficient_model = ValueEfficientEstimator(propensity_model=propensity_model,
                                                outcome_model_lower=outcome_model_lower,
                                                outcome_model_upper=outcome_model_upper,
                                                gamma=cfg.models.gamma_model,
                                                bound_type=cfg.models.policy.sharp_efficient.bound_type,
                                                lr=cfg.models.policy.sharp_efficient.lr)

            efficient_trainer = L.Trainer(max_epochs=cfg.models.policy.sharp_efficient.num_epochs,
                                   callbacks=[efficient_early_stopping])#, logger=mlflow_logger)
            efficient_trainer.fit(efficient_model, policy_train_loader, policy_val_loader)

            ############################################################################################################
            # 4) Evaluate
            efficient_predictions = torch.softmax(efficient_model.policy(torch.tensor(RWdataset.X_test).float()), dim=1).detach().numpy()
            efficient_value = RWdataset.evaluate_policy(efficient_predictions)
            true_value = RWdataset.Y_test.mean()

            regret = efficient_value - true_value

            # Log regret as a metric
            mlflow.log_metric("regret", regret)

if __name__ == "__main__":
    main()

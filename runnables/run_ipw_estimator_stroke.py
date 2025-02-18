import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import numpy as np
import torch
import mlflow
import mlflow.pytorch
from src.nuisance_models import PropensityModel
from src.ipw_estimator import IPWEstimator
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
        mlflow.set_experiment("RWD_IPW")

        with mlflow.start_run():
            # Log parameters under `exp`
            mlflow.log_params(OmegaConf.to_container(cfg.exp, resolve=True))

            # Log parameters under `data`
            mlflow.log_params(OmegaConf.to_container(cfg.data.real_world, resolve=True))

            # Log parameters under `models.policy.doubly_robust`
            mlflow.log_params(OmegaConf.to_container(cfg.models.policy.ipw, resolve=True))


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

            # Propensity model
            prop_early_stopping = EarlyStopping(monitor="val_prop_loss", patience=cfg.exp.patience, mode="min")
            propensity_model = PropensityModel(input_dim=cfg.data.real_world.n_features,
                                               num_treatments=cfg.data.real_world.n_treatments,
                                               lr=cfg.models.nuisance.propensity_model.lr)

            propensity_trainer = L.Trainer(max_epochs=cfg.models.nuisance.propensity_model.num_epochs,
                                           callbacks=[prop_early_stopping])
            propensity_trainer.fit(propensity_model, nuisance_train_loader, nuisance_val_loader)

            ############################################################################################################

            # 3) Policy model

            ipw_early_stopping = EarlyStopping(monitor="val_V", patience=cfg.exp.patience, mode="min")

            ipw_model = IPWEstimator(propensity_model=propensity_model,
                                   lr=cfg.models.policy.doubly_robust.lr)

            ipw_trainer = L.Trainer(max_epochs=cfg.models.policy.doubly_robust.num_epochs,
                                   callbacks=[ipw_early_stopping])#, logger=mlflow_logger)
            ipw_trainer.fit(ipw_model, policy_train_loader, policy_val_loader)


            ############################################################################################################
            # 4) Evaluate
            ipw_predictions = torch.softmax(ipw_model.policy(torch.tensor(RWdataset.X_test).float()), dim=1).detach().numpy()
            ipw_value = RWdataset.evaluate_policy(ipw_predictions)

            regret = ipw_value - RWdataset.Y_test.mean()

            # Log regret as a metric
            mlflow.log_metric("regret", regret)

if __name__ == "__main__":
    main()

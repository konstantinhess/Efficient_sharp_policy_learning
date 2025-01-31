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
from src.plugin_estimator import ValuePlugInEstimator
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
        mlflow.set_experiment("Synthetic_Plugin")
        # mlflow_logger = L.pytorch.loggers.MLFlowLogger(experiment_name="Synthetic_Plugin",
        #                                               tracking_uri=cfg.exp.mlflow_uri)
        with mlflow.start_run():
            # Log parameters under `exp`
            mlflow.log_params(OmegaConf.to_container(cfg.exp, resolve=True))

            # Log parameters under `data`
            mlflow.log_params(OmegaConf.to_container(cfg.data.synthetic, resolve=True))

            # Log parameters under `models.policy.plug_in`
            mlflow.log_params(OmegaConf.to_container(cfg.models.policy.plug_in, resolve=True))


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
            policy_train_loader = create_dataloader(policy_train, batch_size=cfg.data.synthetic.batch_size)
            policy_val_loader = create_dataloader(policy_val, batch_size=cfg.data.synthetic.batch_size)

            ############################################################################################################

            # 2) Nuisance models
            nuisance_early_stopping = EarlyStopping(monitor="val_loss", patience=cfg.exp.patience, mode="min")

            # a) Propensity model
            prop_early_stopping = EarlyStopping(monitor="val_prop_loss", patience=cfg.exp.patience, mode="min")
            propensity_model = PropensityModel(input_dim=cfg.models.nuisance.propensity_model.input_dim,
                                               num_treatments=cfg.models.nuisance.propensity_model.n_treatments,
                                               lr=cfg.models.nuisance.propensity_model.lr)

            propensity_trainer = L.Trainer(max_epochs=cfg.models.nuisance.propensity_model.num_epochs,
                                           callbacks=[prop_early_stopping])#, logger=mlflow_logger)
            propensity_trainer.fit(propensity_model, nuisance_train_loader, nuisance_val_loader)

            # b) Quantile model: alpha^+
            quantile_early_stopping = EarlyStopping(monitor="val_quantile_loss", patience=cfg.exp.patience, mode="min")
            quantile_model_plus = ConditionalQuantileModel(quantile=cfg.models.nuisance.quantile_model.alpha_plus,
                                                           input_dim=cfg.models.nuisance.quantile_model.input_dim,
                                                           n_treatments=cfg.models.nuisance.quantile_model.n_treatments,
                                                           lr=cfg.models.nuisance.quantile_model.lr)

            quantile_plus_trainer = L.Trainer(max_epochs=cfg.models.nuisance.quantile_model.num_epochs,
                                              callbacks=[quantile_early_stopping])#, logger=mlflow_logger)
            quantile_plus_trainer.fit(quantile_model_plus, nuisance_train_loader, nuisance_val_loader)


            # c) Outcome model: \bar{mu}^+
            upper_early_stopping = EarlyStopping(monitor="val_upper_loss", patience=cfg.exp.patience, mode="min")
            outcome_model_upper = OutcomeModel(input_dim=cfg.models.nuisance.outcome_model.input_dim,
                                         num_treatments=cfg.models.nuisance.outcome_model.n_treatments,
                                         type='upper',
                                         quantile=cfg.models.nuisance.quantile_model.alpha_plus,
                                         quantile_model=quantile_model_plus,
                                         lr=cfg.models.nuisance.outcome_model.lr)

            outcome_trainer_upper = L.Trainer(max_epochs=cfg.models.nuisance.outcome_model.num_epochs,
                                        callbacks=[upper_early_stopping])#, logger=mlflow_logger)
            outcome_trainer_upper.fit(outcome_model_upper, nuisance_train_loader, nuisance_val_loader)

            # d) Outcome model: \underbar{mu}+
            lower_early_stopping = EarlyStopping(monitor="val_lower_loss", patience=cfg.exp.patience, mode="min")

            outcome_model_lower = OutcomeModel(input_dim=cfg.models.nuisance.outcome_model.input_dim,
                                             num_treatments=cfg.models.nuisance.outcome_model.n_treatments,
                                             type='lower',
                                             quantile=cfg.models.nuisance.quantile_model.alpha_plus,
                                             quantile_model=quantile_model_plus,
                                             lr=cfg.models.nuisance.outcome_model.lr)

            outcome_trainer_lower = L.Trainer(max_epochs=cfg.models.nuisance.outcome_model.num_epochs,
                                            callbacks=[lower_early_stopping])#, logger=mlflow_logger)
            outcome_trainer_lower.fit(outcome_model_lower, nuisance_train_loader, nuisance_val_loader)

            ############################################################################################################

            # 3) Policy model

            plugin_early_stopping = EarlyStopping(monitor="val_V_bound", patience=cfg.exp.patience, mode="min")

            plugin_model = ValuePlugInEstimator(propensity_model=propensity_model,
                                                outcome_model_lower=outcome_model_lower,
                                                outcome_model_upper=outcome_model_upper,
                                                gamma=cfg.models.policy.plug_in.gamma,
                                                bound_type=cfg.models.policy.plug_in.bound_type,
                                                lr=cfg.models.policy.plug_in.lr)

            plugin_trainer = L.Trainer(max_epochs=cfg.models.policy.plug_in.num_epochs,
                                   callbacks=[plugin_early_stopping])#, logger=mlflow_logger)
            plugin_trainer.fit(plugin_model, policy_train_loader, policy_val_loader)

            ############################################################################################################
            # 4) Evaluate
            plugin_predictions = torch.softmax(plugin_model.policy(torch.tensor(generator_test.X).float()), dim=1).detach().numpy()
            plugin_value = generator_test.evaluate_policy(plugin_predictions)
            randomized_value = generator_test.evaluate_policy(np.ones(plugin_predictions.shape) / plugin_predictions.shape[1])

            regret = plugin_value - randomized_value

            # Log regret as a metric
            mlflow.log_metric("regret", regret)

if __name__ == "__main__":
    main()

from abc import ABC, abstractmethod

import numpy as np
import optuna
import mlflow
import mlflow.sklearn
import xgboost as xgb
import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

optuna.logging.set_verbosity(optuna.logging.ERROR)

# model configs
class BaseModelConfig(ABC):
    name: str

    @abstractmethod
    def suggest_params(self, trial) -> dict: ...

    @abstractmethod
    def build(self, params: dict, seed: int, n_classes: int): ...

class DecisionTreeConfig(BaseModelConfig):
    name = "Decision Tree"

    def suggest_params(self, trial):
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        }

    def build(self, params, seed, n_classes):
        return DecisionTreeClassifier(**params, random_state=seed, class_weight="balanced")

class RandomForestConfig(BaseModelConfig):
    name = "Random Forest"

    def suggest_params(self, trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
            "max_depth": trial.suggest_categorical("max_depth", [None, 10, 20, 30, 40]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }

    def build(self, params, seed, n_classes):
        return RandomForestClassifier(
            **params, random_state=seed, class_weight="balanced", n_jobs=-1)

class XGBoostConfig(BaseModelConfig):
    name = "XGBoost"

    def suggest_params(self, trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
        }

    def build(self, params, seed, n_classes):
        return xgb.XGBClassifier(
            **params, objective="multi:softprob", num_class=n_classes,
            eval_metric="mlogloss", tree_method="hist", random_state=seed, n_jobs=-1)

class LightGBMConfig(BaseModelConfig):
    name = "LightGBM"

    def suggest_params(self, trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
        }

    def build(self, params, seed, n_classes):
        return lgb.LGBMClassifier(
            **params, objective="multiclass", num_class=n_classes,
            class_weight="balanced", subsample_freq=1, random_state=seed, verbose=-1)

# trainer
class ModelTrainer:
    def __init__(self, seed=42, n_trials=30, tune=True):
        self.seed = seed
        self.n_trials = n_trials
        self.tune = tune
        self.model_configs = [
            DecisionTreeConfig(),
            RandomForestConfig(),
            XGBoostConfig(),
            LightGBMConfig(),
        ]

    def _tune(self, config, x_sub, y_sub, x_val, y_val, n_classes):
        def objective(trial):
            model = config.build(config.suggest_params(trial), self.seed, n_classes)
            model.fit(x_sub, y_sub)
            pred = np.argmax(model.predict_proba(x_val), axis=1)
            return f1_score(y_val, pred, average="macro")

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.seed))
        study.optimize(objective, n_trials=self.n_trials)
        print(f"  {config.name}: best val Macro F1 = {study.best_value:.4f}")
        return study.best_params

    def run(self, x_train, y_train):
        mlflow.set_experiment("Credit Score Classification")

        n_classes = len(np.unique(y_train))
        x_sub, x_val, y_sub, y_val = train_test_split(
            x_train, y_train, test_size=0.2,
            random_state=self.seed, stratify=y_train)

        results = {}
        for config in self.model_configs:
            print(f"Tuning + training {config.name} ...")
            params = self._tune(config, x_sub, y_sub, x_val, y_val, n_classes) if self.tune else {}
            model = config.build(params, self.seed, n_classes)

            with mlflow.start_run(run_name=config.name) as run:
                mlflow.log_param("model_name", config.name)
                mlflow.log_params({f"param__{k}": v for k, v in params.items()})
                model.fit(x_train, y_train)
                mlflow.sklearn.log_model(model, artifact_path="model")
                run_id = run.info.run_id

            results[config.name] = {"model": model, "params": params, "run_id": run_id}

        return results

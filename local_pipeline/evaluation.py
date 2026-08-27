import pickle
import time
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

REGISTERED_MODEL_NAME = "credit-score-classifier"

class ModelEvaluator:
    def __init__(self):
        self.output_dir = Path(__file__).parent / "artifacts"

    def _model_size_mb(self, model):
        return len(pickle.dumps(model)) / (1024 ** 2)

    def _latency_1k_ms(self, model, x, n=1000, repeats=5):
        x = np.asarray(x)
        x_1k = (x[:n] if len(x) >= n
                 else x[np.random.default_rng(42).choice(len(x), size=n, replace=True)])
        model.predict(x_1k)
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            model.predict(x_1k)
            times.append((time.perf_counter() - start) * 1000)
        return float(np.median(times))

    def evaluate(self, model, x_test, y_test):
        pred = np.argmax(model.predict_proba(x_test), axis=1)
        return {
            "f1_macro": float(f1_score(y_test, pred, average="macro")),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
            "accuracy": float(accuracy_score(y_test, pred)),
            "model_size_mb": self._model_size_mb(model),
            "latency_1k_ms": self._latency_1k_ms(model, x_test),
        }

    def evaluate_all(self, results, x_test, y_test):
        for name, entry in results.items():
            metrics = self.evaluate(entry["model"], x_test, y_test)
            with mlflow.start_run(run_id=entry["run_id"]):
                mlflow.log_metrics(metrics)
            results[name]["metrics"] = metrics
            print(f"  {name:20} f1_macro={metrics['f1_macro']:.4f}  bal_acc={metrics['balanced_accuracy']:.4f}"
                  f"  acc={metrics['accuracy']:.4f}  size={metrics['model_size_mb']:.3f}MB  lat={metrics['latency_1k_ms']:.0f}ms")
        return results

    def select_best(self, results):
        table = pd.DataFrame({name: results[name]["metrics"] for name in results}).T.astype(float)
        table["f1_macro_r"] = table["f1_macro"].round(4)
        table["balanced_accuracy_r"] = table["balanced_accuracy"].round(4)
        table["accuracy_r"] = table["accuracy"].round(4)
        return table.sort_values(
            by=["f1_macro_r", "balanced_accuracy_r", "accuracy_r", "model_size_mb", "latency_1k_ms"],
            ascending=[False, False, False, True, True]).index[0]

    def save(self, preprocessor, best_name, results, classes):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        bundle = Pipeline([("preprocess", preprocessor), ("model", results[best_name]["model"])])
        joblib.dump({"pipeline": bundle, "classes": classes},
                    self.output_dir / "credit_score_pipeline.pkl")
        with mlflow.start_run(run_id=results[best_name]["run_id"]):
            mlflow.sklearn.log_model(bundle, artifact_path="model", registered_model_name=REGISTERED_MODEL_NAME)
        print(f"  saved + registered {REGISTERED_MODEL_NAME} -> {self.output_dir.name}/")

    def run(self, preprocessor, results, classes, x_test, y_test):
        results = self.evaluate_all(results, x_test, y_test)
        best_name = self.select_best(results)
        print(f"  -> {best_name}")
        self.save(preprocessor, best_name, results, classes)
        return best_name, results[best_name]["metrics"]["f1_macro"]

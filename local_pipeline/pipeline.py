import warnings
from pathlib import Path

from data_ingestion import DataIngestion
from preprocessing import CreditPreprocessor
from train import ModelTrainer
from evaluation import ModelEvaluator

warnings.filterwarnings("ignore")

SEED = 42
F1_THRESHOLD = 0.65

class CreditScorePipeline:
    def __init__(self, seed=SEED, tune=True, f1_threshold=F1_THRESHOLD):
        self.seed = seed
        self.f1_threshold = f1_threshold
        base_dir = Path(__file__).parent
        self.ingestor = DataIngestion(base_dir.parent / "data" / "credit_score.csv", base_dir / "ingested")
        self.preprocessor = CreditPreprocessor()
        self.trainer = ModelTrainer(seed=seed, tune=tune)
        self.evaluator = ModelEvaluator()

    def run(self):
        data_file = self.ingestor.run()
        x_train, x_test, y_train, y_test, classes = self.preprocessor.run(data_file, self.seed)
        results = self.trainer.run(x_train, y_train)
        best_name, f1 = self.evaluator.run(self.preprocessor, results, classes, x_test, y_test)

        if f1 >= self.f1_threshold:
            print(f"APPROVED - {best_name}  f1_macro={f1:.4f}")
        else:
            print(f"REJECTED - {best_name}  f1_macro={f1:.4f} < threshold={self.f1_threshold}")

        return best_name

if __name__ == "__main__":
    CreditScorePipeline().run()

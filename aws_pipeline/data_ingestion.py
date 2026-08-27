import os
import pandas as pd

class DataIngestion:
    def __init__(self):
        if os.environ.get("SM_CHANNEL_TRAIN") or os.path.exists("/opt/ml/processing"):
            source_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/processing/input")
            self.output_dir = "/opt/ml/processing/ingested"
        else:
            base_dir = os.path.dirname(__file__)
            source_dir = os.path.join(base_dir, "..", "data")
            self.output_dir = os.path.join(base_dir, "ingested")
        self.source_file = os.path.join(source_dir, "credit_score.csv")
        self.output_file = os.path.join(self.output_dir, "credit_score.csv")

    def run(self):
        os.makedirs(self.output_dir, exist_ok=True)
        if not os.path.exists(self.source_file):
            raise ValueError(f"Dataset not found: {self.source_file}")
        df = pd.read_csv(self.source_file)
        if df.empty:
            raise ValueError("Dataset is empty")
        df.to_csv(self.output_file, index=False)
        print(f"Data ingested: {self.source_file} -> {self.output_file}")
        return self.output_file

if __name__ == "__main__":
    DataIngestion().run()

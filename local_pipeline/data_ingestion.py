from pathlib import Path
import pandas as pd

class DataIngestion:
    def __init__(self, source_file, output_dir):
        self.source_file = Path(source_file)
        self.output_dir = Path(output_dir)
        self.output_file = self.output_dir / "credit_score.csv"

    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(self.source_file)
        if df.empty:
            raise ValueError("Dataset is empty")
        df.to_csv(self.output_file, index=False)
        print(f"Data ingested: {self.source_file} -> {self.output_file}")
        return self.output_file

if __name__ == "__main__":
    base_dir = Path(__file__).parent
    DataIngestion(base_dir.parent / "data" / "credit_score.csv", base_dir / "ingested").run()

# Credit Score Classification: Local and AWS Deployment

An end-to-end machine learning system for multiclass credit-score classification. The repository includes reproducible local training and evaluation, an AWS SageMaker inference path, a Streamlit interface, and deployment utilities.

## Project structure

```text
data/
  credit_score.csv
local_pipeline/
  data_ingestion.py
  preprocessing.py
  train.py
  evaluation.py
  inference.py
  pipeline.py
  app.py
aws_pipeline/
  data_ingestion.py
  preprocessing.py
  train.py
  evaluation.py
  inference.py
  pipeline.py
  app.py
  requirements.txt
notebooks/
  deploy_endpoint.ipynb
deploy/
  ec2_user_data.sh
```

## Local workflow

```bash
cd local_pipeline
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python pipeline.py
streamlit run app.py
```

The local pipeline ingests the data, cleans mixed-type financial fields, handles implausible values, engineers loan and calendar features, compares Decision Tree, Random Forest, XGBoost, and LightGBM models with Optuna, logs experiments to MLflow, evaluates macro F1/accuracy/latency/model size, and saves the selected pipeline to `local_pipeline/artifacts/`.

## AWS workflow

The AWS path mirrors the preprocessing and model contract used locally. It can train in SageMaker processing/training jobs, package the model artifact, deploy a real-time endpoint, and serve predictions through the Streamlit app.

1. Run the local pipeline or the AWS training pipeline to create `model/model.joblib`.
2. Package it as `model.tar.gz` with the deployment notebook.
3. Set `SAGEMAKER_BUCKET`, `AWS_REGION`, and `SAGEMAKER_ROLE_NAME` in your environment.
4. Run `notebooks/deploy_endpoint.ipynb` to upload and deploy the model.
5. Run `streamlit run aws_pipeline/app.py` with `ENDPOINT_NAME` configured.

The endpoint expects JSON shaped as `{"instances": [{...features...}]}` and returns class labels, class IDs, and probabilities for `Poor`, `Standard`, and `Good`.

## Notes

- Generated artifacts such as `mlruns/`, `mlflow.db`, model checkpoints, and `model.tar.gz` are excluded from version control. They can be regenerated from the included scripts.
- The included CSV is used for experimentation. Confirm its license and provenance before redistributing the repository.
- Do not commit AWS credentials, private customer data, or account-specific identifiers.

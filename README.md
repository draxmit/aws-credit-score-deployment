# Credit Score Classification with AWS SageMaker

A Streamlit inference client for a multiclass credit-score model hosted behind an AWS SageMaker real-time endpoint.

## What it covers

- A form-based interface for credit-profile inputs
- JSON request construction for a SageMaker runtime endpoint
- Prediction labels and class probabilities
- Runtime configuration through `ENDPOINT_NAME` and `AWS_REGION`
- Error handling for missing credentials and AWS client errors

## Architecture

```text
Streamlit UI -> boto3 SageMaker Runtime -> deployed model endpoint
                                      -> label + class probabilities
```

This repository contains the client application. Model training, packaging, and SageMaker endpoint provisioning must be completed separately in AWS.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
$env:ENDPOINT_NAME = "credit-score-endpoint"
$env:AWS_REGION = "us-east-1"
streamlit run app.py
```

Configure AWS credentials using an IAM role or the AWS CLI before making predictions. Never commit credentials, endpoint secrets, or private customer data.

## Expected endpoint response

The endpoint should return JSON with `labels` and `probabilities` arrays. The client expects three classes: `Poor`, `Standard`, and `Good`.

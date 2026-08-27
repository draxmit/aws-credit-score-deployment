import os
import json

import joblib
import numpy as np
import pandas as pd

JSON_CONTENT_TYPE = "application/json"
CLASS_NAMES = ["Poor", "Standard", "Good"]

def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "model.joblib"))["pipeline"]

def input_fn(request_body, request_content_type):
    if request_content_type == JSON_CONTENT_TYPE:
        payload = json.loads(request_body)
        instances = payload["instances"]
        return pd.DataFrame(instances)
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, pipeline):
    proba = pipeline.predict_proba(input_data)
    class_ids = np.argmax(proba, axis=1)
    return {
        "probabilities": proba.tolist(),
        "predictions": class_ids.tolist(),
        "labels": [CLASS_NAMES[int(i)] for i in class_ids],
    }

def output_fn(prediction, accept_content_type):
    if accept_content_type == JSON_CONTENT_TYPE:
        return json.dumps(prediction), JSON_CONTENT_TYPE
    raise ValueError(f"Unsupported accept type: {accept_content_type}")

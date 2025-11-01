# train.py
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from google.cloud import bigquery, storage

# --- Configuration ---
PROJECT_ID = os.environ.get("PROJECT_ID")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
MODEL_DIR = "models/iris/"

# --- 1. Load Data ---
print("Loading data from BigQuery...")
client = bigquery.Client(project=PROJECT_ID)
query = """
    SELECT 
        sepal_length, sepal_width, petal_length, petal_width, 
        CAST(species AS STRING) as species
    FROM `bigquery-public-data.ml_datasets.iris`
"""
df = client.query(query).to_dataframe()

# --- 2. Prepare Data ---
print("Preparing data...")
# Simple label encoding
df['species'] = df['species'].astype('category').cat.codes
X = df.drop(columns=['species'])
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. Train Model ---
print("Training model...")
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# --- 4. Evaluate Model ---
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {acc}")

# --- 5. Save Model ---
print("Saving model...")
model_filename = 'model.joblib'

# If AIP_MODEL_DIR is set, save the model to that directory.
# This is the case when running on Vertex AI.
if "AIP_MODEL_DIR" in os.environ:
    output_dir = os.environ["AIP_MODEL_DIR"]
    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(output_dir, model_filename))
    print(f"Model saved to {output_dir}/{model_filename}")
else:
    # Otherwise, save to GCS (for local testing or other environments)
    joblib.dump(model, model_filename)
    gcs_client = storage.Client()
    bucket = gcs_client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"{MODEL_DIR}{model_filename}")
    blob.upload_from_filename(model_filename)
    print(f"Model saved to gs://{BUCKET_NAME}/{MODEL_DIR}{model_filename}")
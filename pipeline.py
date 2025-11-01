# pipeline.py
import kfp
from kfp.v2 import dsl
from kfp.v2.compiler import Compiler
from google_cloud_pipeline_components.v1.model import ModelUploadOp
from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp

# --- Configuration ---
# You must change these values
PROJECT_ID = "nextgcp-473616"  # <-- CHANGE THIS
BUCKET_NAME = " my-mlops-project-bucket" # <-- CHANGE THIS
REGION = "us-central1"
PIPELINE_ROOT = f"gs://{BUCKET_NAME}/pipeline-root/"
TRAINER_IMAGE_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/ml-models/iris-trainer:latest"
MODEL_DISPLAY_NAME = "iris-classifier"

@dsl.pipeline(
    name="iris-classification-pipeline",
    description="Trains and registers an Iris classification model.",
    pipeline_root=PIPELINE_ROOT,
)
def pipeline(
    project: str = PROJECT_ID,
    bucket: str = BUCKET_NAME
):
    # --- Step 1: Train the model ---
    # Runs the Docker container as a Vertex AI Custom Job
    train_op = CustomTrainingJobOp(
        project=project,
        display_name="train-iris-model",
        worker_pool_specs=[
            {
                "machine_spec": {"machine_type": "n1-standard-4"},
                "replica_count": 1,
                "container_spec": {
                    "image_uri": TRAINER_IMAGE_URI,
                    # Pass environment variables to train.py
                    "env": [
                        {"name": "PROJECT_ID", "value": project},
                        {"name": "BUCKET_NAME", "value": bucket},
                    ],
                },
            }
        ],
    )

    # --- Step 2: Upload the model to Vertex AI Model Registry ---
    # The training job logs the GCS path of the model.
    # We retrieve that log and pass it to the ModelUploadOp.
    
    # This is a bit advanced: it parses the output log from the training job
    # to find the line that starts with "gcs_path:"
    model_gcs_path = train_op.outputs["gcs_output_directory"]

    upload_op = ModelUploadOp(
        project=project,
        display_name=MODEL_DISPLAY_NAME,
        artifact_uri=model_gcs_path,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest",
    )
    
    # This ensures the upload runs *after* the training
    upload_op.after(train_op)

# --- Compile the pipeline ---
if __name__ == "__main__":
    Compiler().compile(
        pipeline_func=pipeline,
        package_path="iris_pipeline.json",
    )
    print("Pipeline compiled to iris_pipeline.json")
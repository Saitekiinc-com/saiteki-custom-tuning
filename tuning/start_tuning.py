import os
import json
import urllib.request
import urllib.error
import sys
import subprocess

# User provided info
PROJECT_ID = "gen-lang-client-0646195883"

# Fetch Access Token via gcloud
try:
    print("Fetching access token via gcloud...")
    # Using relative path to the installed gcloud (one level up)
    ACCESS_TOKEN = subprocess.check_output(
        ["../google-cloud-sdk/bin/gcloud", "auth", "print-access-token"],  
        text=True
    ).strip()
except Exception as e:
    print(f"Failed to get access token: {e}")
    sys.exit(1)

# Derived from https://storage.googleapis.com/saiteki-model/training_data_vertex.jsonl
TRAINING_DATA_URI = "gs://saiteki-model/training_data_vertex.jsonl" 
# Note: For local upload, it would be "tuning/data/training.jsonl" relative to project root, 
# but this script uses GCS URI so no change needed for execution if using GCS.
# However, if using local file upload logic (not shown in snippet but implied as option), path would change.
# The original code uses TRAINING_DATA_URI which is GCS, so actually no change strictly needed for GCS mode.
# But let's check if the script tries to open a local file.
REGION = "us-central1"

def tune_vertex():
    # Documentation: https://cloud.google.com/vertex-ai/docs/reference/rest/v1beta1/projects.locations.tuningJobs/create
    url = f"https://{REGION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT_ID}/locations/{REGION}/tuningJobs"
    
    payload = {
        "tunedModelDisplayName": "saiteki-fine-tuning-01",
        "supervisedTuningSpec": {
            "trainingDatasetUri": TRAINING_DATA_URI,
            "validationDatasetUri": None,
            "hyperParameters": {
                "epochCount": 5,
                "learningRateMultiplier": 1.0
            }
        },
        "baseModel": "gemini-3.0-flash-001" 
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ACCESS_TOKEN}"
    }
    data = json.dumps(payload).encode("utf-8")
    
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    
    print(f"Submitting Tuning Job to Vertex AI ({REGION})...")
    print(f"Payload: {json.dumps(payload, indent=2)}")

    try:
        with urllib.request.urlopen(req) as response:
            resp_body = response.read().decode("utf-8")
            print("Job Submitted Successfully!")
            print(resp_body)
            
            resp_json = json.loads(resp_body)
            job_name = resp_json.get("name", "Unknown")
            print(f"\nTrack status with URL: https://{REGION}-aiplatform.googleapis.com/v1/{job_name}?key={API_KEY}")

    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} {e.reason}")
        print(e.read().decode("utf-8"))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    tune_vertex()

import subprocess
import urllib.request
import json
import sys

PROJECT_ID = "gen-lang-client-0646195883"
REGION = "us-central1"
JOB_ID = "8139917865869377536"

try:
    token = subprocess.check_output(["./google-cloud-sdk/bin/gcloud", "auth", "print-access-token"], text=True).strip()
    
    url = f"https://{REGION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT_ID}/locations/{REGION}/tuningJobs/{JOB_ID}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    
    with urllib.request.urlopen(req) as res:
        data = json.loads(res.read())
        print(json.dumps(data, indent=2))
        
        state = data.get("state")
        tuned_model = data.get("tunedModel")
        
        print(f"\nState: {state}")
        if tuned_model:
            print(f"Tuned Model ID: {tuned_model.get('model')}") # Structure might vary
        
except Exception as e:
    print(f"Error: {e}")

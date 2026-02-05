import os
import json
import urllib.request
import urllib.error
import sys

# --- Configuration from Environment Variables ---
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
REGION = os.environ.get("GCP_REGION", "us-central1")
VERTEX_API_KEY = os.environ.get("VERTEX_API_KEY")
VERTEX_ENDPOINT_ID = os.environ.get("VERTEX_ENDPOINT_ID")
BASE_MODEL_ID = "gemini-2.0-flash-001"

# Default Prompt if not provided via args
DEFAULT_PROMPT = "部下から『モチベーションが上がらない』と相談されました。どう対応しますか？"

def call_api(model_resource_url, prompt, label):
    url = f"{model_resource_url}:streamGenerateContent?key={VERTEX_API_KEY}"
    
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 800, "temperature": 0.7}
    }
    
    headers = {"Content-Type": "application/json"}
    data = json.dumps(payload).encode("utf-8")
    
    full_text = ""
    try:
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req) as response:
            full_resp = response.read().decode("utf-8")
            try:
                resp_list = json.loads(full_resp)
                for item in resp_list:
                    if "candidates" in item:
                        for cand in item["candidates"]:
                             if "content" in cand and "parts" in cand["content"]:
                                 for part in cand["content"]["parts"]:
                                     full_text += part.get("text", "")
            except Exception as e:
                return f"Error parsing JSON: {e}"
    except Exception as e:
        return f"Error ({label}): {e}"
    
    return full_text

def main():
    if not VERTEX_API_KEY:
        print("Error: VERTEX_API_KEY environment variable not set.")
        sys.exit(1)

    prompt = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PROMPT
    
    print(f"Prompt: {prompt}\n")
    
    # 1. Base Model
    base_url = f"https://{REGION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT_ID}/locations/{REGION}/publishers/google/models/{BASE_MODEL_ID}"
    base_res = call_api(base_url, prompt, "Base Model")
    
    # 2. Tuned Model
    tuned_url = f"https://{REGION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/{VERTEX_ENDPOINT_ID}"
    tuned_res = call_api(tuned_url, prompt, "Tuned Model")
    
    # Output in Markdown format for GitHub Issue
    print("## Model Comparison Report")
    print(f"**Prompt**: `{prompt}`\n")
    
    print("| Model | Response |")
    print("| :--- | :--- |")
    # Escape newlines for table format or use blockquote
    # For better readability in Issue, let's use details/summary instead of a huge table
    
    print(f"### 🔹 Base Model ({BASE_MODEL_ID})")
    print(base_res)
    print("\n---\n")
    
    print(f"### 🔸 Tuned Model (Fine-Tuned)")
    print(tuned_res)

if __name__ == "__main__":
    main()

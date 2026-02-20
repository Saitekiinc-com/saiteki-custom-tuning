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
        "generationConfig": {"maxOutputTokens": 8192, "temperature": 0.7}
    }
    
    headers = {"Content-Type": "application/json"}
    data = json.dumps(payload).encode("utf-8")
    
    print(f"DEBUG [{label}]: Sending Payload: {json.dumps(payload, ensure_ascii=False)}")
    
    full_text = ""
    try:
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req) as response:
            full_resp = response.read().decode("utf-8")
            print(f"DEBUG [{label}]: Received Response: {full_resp}")
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

import argparse
import re

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run specific model comparison tasks.")
    parser.add_argument("prompt", nargs="?", help="The prompt to send to the models.")
    parser.add_argument("--mode", choices=["parse", "base", "tuned", "simultaneous"], default="simultaneous", help="Execution mode.")
    parser.add_argument("--body", help="Issue body content for parsing prompt.")
    return parser.parse_args()

def main():
    args = parse_arguments()

    # --- Mode: Parse Prompt from Issue Body ---
    if args.mode == "parse":
        content = args.body or os.environ.get("ISSUE_BODY")
        if not content:
             print("Error: Body content is required (via --body or ISSUE_BODY env var).")
             sys.exit(1)
        
        # Simple parsing for Issue Form (assuming formatting or direct text)
        # Issue Forms usually return key-value pairs or just sections.
        # We'll just take the body as is if it's simple, or try to find the prompt section.
        # For simplicity in this v1, assuming the user writes the prompt or the template logic works.
        # Actually, GitHub Issue Forms (yaml) put the content in the body.
        # If using the YAML template provided, the body will contain:
        # ### Prompt
        #
        # <user input>
        
        # content is already set above
        match = re.search(r"### Prompt\s*\n\s*(.*)", content, re.DOTALL)
        if match:
            print(match.group(1).strip())
        else:
            # Fallback: just return the whole body if pattern not found
            print(content.strip())
        return

    # --- Standard Execution ---
    if not VERTEX_API_KEY:
        print("Error: VERTEX_API_KEY environment variable not set.")
        sys.exit(1)

    prompt = args.prompt
    if not prompt:
        print("Error: prompt argument is required for model execution modes.")
        sys.exit(1)

    # 1. Base Model
    if args.mode in ["base", "simultaneous"]:
        base_url = f"https://{REGION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT_ID}/locations/{REGION}/publishers/google/models/{BASE_MODEL_ID}"
        base_res = call_api(base_url, prompt, "Base Model")
        
        print(f"### 🔹 Base Model ({BASE_MODEL_ID})")
        print(base_res)

    # 2. Tuned Model
    if args.mode in ["tuned", "simultaneous"]:
        tuned_url = f"https://{REGION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/{VERTEX_ENDPOINT_ID}"
        tuned_res = call_api(tuned_url, prompt, "Tuned Model")
        
        print(f"### 🔸 Tuned Model (Fine-Tuned)")
        print(tuned_res)

if __name__ == "__main__":
    main()

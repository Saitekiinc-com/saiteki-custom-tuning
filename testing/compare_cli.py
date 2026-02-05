import json
import urllib.request
import urllib.error

PROJECT_ID = "gen-lang-client-0646195883"
REGION = "us-central1"
API_KEY = "AQ.Ab8RN6Kx86NMYSqVQfPivmZ26P7MjS3i6THSEyjKP7fEFmlHfA" 

# Target Prompt
PROMPT = "仕事終わりに部下から日報が提出されましたが、内容が薄く具体性に欠けており、その日の動きや課題がよく分かりません。あなたならどのように指摘・フィードバックをしますか？"

def call_api(model_resource_url, label):
    print(f"\n[{label}]")
    print("-" * 40)
    
    url = f"{model_resource_url}:streamGenerateContent?key={API_KEY}"
    
    payload = {
        "contents": [{"role": "user", "parts": [{"text": PROMPT}]}],
        "generationConfig": {"maxOutputTokens": 1000, "temperature": 0.7}
    }
    
    headers = {"Content-Type": "application/json"}
    data = json.dumps(payload).encode("utf-8")
    
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
                                     print(part["text"], end="")
            except:
                pass
            print("\n")
    except Exception as e:
        print(f"Error: {e}")

# 1. Base Model (gemini-2.0-flash-001)
base_url = f"https://{REGION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT_ID}/locations/{REGION}/publishers/google/models/gemini-2.0-flash-001"
call_api(base_url, "Base Model (Untuned)")

# 2. Tuned Model (Endpoint)
endpoint_url = f"https://{REGION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/3814882036406026240"
call_api(endpoint_url, "Tuned Model (Fine-Tuned)")

import json
import urllib.request
import urllib.error

PROJECT_ID = "gen-lang-client-0646195883"
ENDPOINT_ID = "3814882036406026240"
REGION = "us-central1"
# User's API Key
API_KEY = "AQ.Ab8RN6Kx86NMYSqVQfPivmZ26P7MjS3i6THSEyjKP7fEFmlHfA" 

def ask_model(label, prompt):
    url = f"https://{REGION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}:streamGenerateContent?key={API_KEY}"
    
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 500, "temperature": 0.7}
    }
    
    headers = {"Content-Type": "application/json"}
    data = json.dumps(payload).encode("utf-8")
    
    print(f"\n[{label} Pattern]: {prompt}")
    print("-" * 40)
    
    try:
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req) as response:
            full_resp = response.read().decode("utf-8")
            # Parse the full JSON response (it acts like a list of responses in one go usually if not streamed line by line correctly in urllib)
            # But streamGenerateContent returns a list of JSON objects wrapped in square brackets.
            try:
                resp_list = json.loads(full_resp)
                for item in resp_list:
                    if "candidates" in item:
                        for cand in item["candidates"]:
                             if "content" in cand and "parts" in cand["content"]:
                                 for part in cand["content"]["parts"]:
                                     print(part["text"], end="")
            except:
                print("Raw output (parsing failed):", full_resp[:200])
            print("\n")
    except Exception as e:
        print(f"Error: {e}")

# Pattern 1: Normal (Business/Persona relevant)
# Context: Dealing with ambiguous instructions or trouble (Training data domain)
ask_model("正常 (Business Persona)", "部下から『モチベーションが上がらない』と相談されました。どう対応しますか？")

# Pattern 2: Abnormal (Irrelevant domain)
# ask_model("異常 (Irrelevant Domain)", "美味しいオムライスの作り方を教えて")

# Pattern 3: Feedback on Bad Daily Report (User Request)
ask_model("日報指摘 (Feedback)", "仕事終わりに部下から日報が提出されましたが、内容が薄く具体性に欠けており、その日の動きや課題がよく分かりません。あなたならどのように指摘・フィードバックをしますか？")

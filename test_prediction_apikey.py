import json
import urllib.request
import urllib.error

PROJECT_ID = "gen-lang-client-0646195883"
API_KEY = "AQ.Ab8RN6Kx86NMYSqVQfPivmZ26P7MjS3i6THSEyjKP7fEFmlHfA" 
REGION = "us-central1"
MODEL_ID = "gemini-2.0-flash-001"

def test_prediction():
    # URL for generating content via Vertex AI (using API Key)
    # Ref: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/gemini#request-body
    url = f"https://{REGION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT_ID}/locations/{REGION}/publishers/google/models/{MODEL_ID}:streamGenerateContent?key={API_KEY}"
    
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "Hello, this is a test using Vertex AI API Key. Can you reply?"}]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 100
        }
    }
    
    headers = {"Content-Type": "application/json"}
    data = json.dumps(payload).encode("utf-8")
    
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    
    print(f"Testing Prediction with API Key on {MODEL_ID}...")
    try:
        with urllib.request.urlopen(req) as response:
            print("Success! Response:")
            while True:
                line = response.readline()
                if not line: break
                print(line.decode("utf-8").strip())
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} {e.reason}")
        print(e.read().decode("utf-8"))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_prediction()

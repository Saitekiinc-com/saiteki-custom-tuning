import json
import urllib.request
import urllib.error

PROJECT_ID = "gen-lang-client-0646195883"
API_KEY = "AQ.Ab8RN6Kx86NMYSqVQfPivmZ26P7MjS3i6THSEyjKP7fEFmlHfA" 
REGION = "us-central1"

# The Endpoint ID found in the job status (Checkpoint 5)
ENDPOINT_ID = "3814882036406026240"
# Resource name: projects/630460208058/locations/us-central1/endpoints/3814882036406026240

def test_tuned_prediction():
    # URL for generating content via Vertex AI Endpoint
    url = f"https://{REGION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}:streamGenerateContent?key={API_KEY}"
    
    print(f"Testing Prediction with ENDPOINT {ENDPOINT_ID}...")

    # --- 利用パターン1: 単発の質問 (Simple Prompt) ---
    print(f"\n--- Pattern 1: Simple Prompt ---")
    payload_simple = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "納期が絶対の案件でトラブル発生。品質を落とせば間に合うが、どうする？"}]
            }
        ],
        "generationConfig": {"maxOutputTokens": 300}
    }
    call_api(url, payload_simple)

    # --- 利用パターン2: コンテキスト/会話履歴を含める (Context / Chat History) ---
    # 過去のやり取りを含めることで、文脈（コンテキスト）を踏まえた回答が得られます。
    print(f"\n--- Pattern 2: With Context (Chat History) ---")
    payload_context = {
        "contents": [
            # 1ターン目: ユーザー
            {"role": "user", "parts": [{"text": "あなたはどのようなスタンスで働いていますか？"}]},
            # 1ターン目: モデルの回答 (履歴として渡す)
            {"role": "model", "parts": [{"text": "私は常にパートナーとして、依頼主のビジネス成功を第一に考え、単なる作業者ではなく提案型の動きを心がけています。"}]},
            # 2ターン目: 今回の質問 (前提を踏まえて回答してくれる)
            {"role": "user", "parts": [{"text": "では、理不尽な仕様変更が来たときはどうしますか？"}]}
        ],
        "generationConfig": {"maxOutputTokens": 300}
    }
    call_api(url, payload_context)

def call_api(url, payload):
    headers = {"Content-Type": "application/json"}
    data = json.dumps(payload).encode("utf-8")
    
    try:
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req) as response:
            # print("Response:")
            while True:
                line = response.readline()
                if not line: break
                # Simple parsing for stream format
                try:
                    line_text = line.decode("utf-8").strip()
                    if line_text: print(line_text[:100] + "..." if len(line_text)>100 else line_text)
                except: pass
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} {e.reason}")
        # print(e.read().decode("utf-8"))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    call_api = test_tuned_prediction()

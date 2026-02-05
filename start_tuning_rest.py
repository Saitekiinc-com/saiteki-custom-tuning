import os
import json
import urllib.request
import urllib.error
import sys

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    print("Error: GEMINI_API_KEY not set")
    sys.exit(1)

def run_tuning():
    # Read messages from JSONL
    examples = []
    with open("training_data.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    
    print(f"Loaded {len(examples)} examples.")
    
    # Convert 'messages' to 'text_input' and 'output' for the API
    formatted_examples = []
    for ex in examples:
        msgs = ex.get('messages', [])
        if len(msgs) == 2:
            formatted_examples.append({
                "text_input": msgs[0]['content'],
                "output": msgs[1]['content']
            })
    
    # Duplicate examples to meet batch size requirements if needed
    while len(formatted_examples) < 4:
         formatted_examples.extend(formatted_examples)
    # Trim to 4 slightly for neatness if we overshot (though extend doubles it so 2->4 is exact)
    
    print(f"Expanded to {len(formatted_examples)} examples for batch size constraint.")
    
    payload = {
        "display_name": "Saiteki Fine-Tuning 01",
        "base_model": "models/gemini-1.0-pro-001", 
        "tuning_task": {
            "hyperparameters": {
                "epoch_count": 5,
                "batch_size": 4,
                "learning_rate": 0.001
            },
            "training_data": {
                "examples": {
                    "examples": formatted_examples
                }
            }
        }
    }

    url = f"https://generativelanguage.googleapis.com/v1beta/tunedModels?key={API_KEY}"
    headers = {"Content-Type": "application/json"}
    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    try:
        print("Sending request to Gemini API (v1beta)...")
        with urllib.request.urlopen(req) as response:
            resp_body = response.read().decode("utf-8")
            print("Success!")
            print(resp_body)
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} {e.reason}")
        print(e.read().decode("utf-8"))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_tuning()

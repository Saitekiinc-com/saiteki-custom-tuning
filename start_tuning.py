import os
import sys
import json
from google import genai
from google.genai import types

# API Key provided by user
API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    print("Error: GEMINI_API_KEY environment variable not set.")
    sys.exit(1)

client = genai.Client(api_key=API_KEY)

def upload_and_tune():
    file_path = "training_data.jsonl"
    display_name = "Saiteki Model Fine-tuning 01"

    print(f"Loading data from {file_path}...")
    training_data = []
    # Read the JSONL file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    training_data.append(json.loads(line))
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    print(f"Loaded {len(training_data)} examples.")

    try:
        print("Starting tuning job with google-genai SDK...")
        
        # In the new SDK, 'tune' is a convenience method.
        # We need to make sure the dataset format is compatible (list of dicts with 'messages').
        
        # Prepare dataset as a dictionary matching the expected structure
        dataset = {"examples": training_data}

        operation = client.tunings.tune(
            base_model="models/gemini-1.5-flash-001-tuning",
            training_dataset=dataset,
            config=types.CreateTuningJobConfig(
                epoch_count=5,
                batch_size=1,
                learning_rate=0.001,
            )
        )
        
        print(f"Tuning job started successfully!")
        print(f"Name: {operation.name}")
        print("This runs asynchronously on Google's side.")
        
        return operation.name

    except Exception as e:
        print(f"An error occurred: {e}")
        if hasattr(e, 'message'):
           print(e.message)

if __name__ == "__main__":
    if not os.path.exists("training_data.jsonl"):
        print("training_data.jsonl not found.")
        sys.exit(1)
        
    upload_and_tune()

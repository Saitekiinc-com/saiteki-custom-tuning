import os
import google.generativeai as genai

API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

models_to_check = [
    "models/gemini-1.5-flash-001",
    "models/gemini-1.5-flash-001-tuning",
    "models/gemini-1.0-pro-001",
    "models/gemini-1.0-pro-001-tuning"
]

print("Checking specific models:")
for m in genai.list_models():
    if m.name in models_to_check or "tuning" in m.name:
        print(f"- {m.name}")
        print(f"  Methods: {m.supported_generation_methods}")

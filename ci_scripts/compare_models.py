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

import argparse
import re

# --- 定量評価ロジック ---

def evaluate_response(text):
    """テキストから定量指標を計算して返す"""
    char_count = len(text)
    paragraphs = [p for p in text.split("\n") if p.strip()]
    paragraph_count = len(paragraphs)
    effective_char_count = len(text.replace(" ", "").replace("　", "").replace("\n", "").replace("\r", ""))
    return {
        "char_count": char_count,
        "paragraph_count": paragraph_count,
        "effective_char_count": effective_char_count,
    }

def format_score_report(base_score, tuned_score):
    """両スコアを比較してMarkdown形式の評価サマリーを返す"""
    rows = [
        ("文字数", base_score["char_count"], tuned_score["char_count"]),
        ("段落数", base_score["paragraph_count"], tuned_score["paragraph_count"]),
        ("実質文字数", base_score["effective_char_count"], tuned_score["effective_char_count"]),
    ]

    table_lines = [
        "| 指標 | 🔹 ベースモデル | 🔸 チューニング済み | 差分 |",
        "|---|---|---|---|",
    ]
    win_count = 0
    for label, base_val, tuned_val in rows:
        diff = tuned_val - base_val
        sign = "+" if diff >= 0 else ""
        mark = "✅" if diff > 0 else ("➖" if diff == 0 else "⚠️")
        if diff > 0:
            win_count += 1
        table_lines.append(f"| {label} | {base_val} | {tuned_val} | {sign}{diff} {mark} |")

    if win_count == len(rows):
        verdict = "**判定:** チューニング済みモデルの回答がすべての指標で上回っています。"
    elif win_count > 0:
        verdict = f"**判定:** チューニング済みモデルが {win_count}/{len(rows)} 指標で上回っています。"
    else:
        verdict = "**判定:** ベースモデルと同等かそれ以上の結果です。チューニングデータの見直しを検討してください。"

    report = "## 📊 定量評価サマリー\n\n"
    report += "\n".join(table_lines)
    report += f"\n\n{verdict}\n"
    return report

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run specific model comparison tasks.")
    parser.add_argument("prompt", nargs="?", help="The prompt to send to the models.")
    parser.add_argument("--mode", choices=["parse", "base", "tuned", "simultaneous", "evaluate"], default="simultaneous", help="Execution mode.")
    parser.add_argument("--body", help="Issue body content for parsing prompt.")
    parser.add_argument("--base-file", help="Base model result file path (for evaluate mode).")
    parser.add_argument("--tuned-file", help="Tuned model result file path (for evaluate mode).")
    parser.add_argument("--output", default="score_result.md", help="Output file path for evaluate mode.")
    return parser.parse_args()

def main():
    args = parse_arguments()

    # --- Mode: Evaluate (定量評価のみ) ---
    if args.mode == "evaluate":
        base_file = args.base_file
        tuned_file = args.tuned_file
        if not base_file or not tuned_file:
            print("Error: --base-file と --tuned-file が必要です。")
            sys.exit(1)
        with open(base_file, "r", encoding="utf-8") as f:
            base_text = f.read()
        with open(tuned_file, "r", encoding="utf-8") as f:
            tuned_text = f.read()
        base_score = evaluate_response(base_text)
        tuned_score = evaluate_response(tuned_text)
        report = format_score_report(base_score, tuned_score)
        output_path = args.output
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(report)
        return

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

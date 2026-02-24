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
BASE_MODEL_ID = "gemini-2.5-flash"

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

# --- LLM-as-Judge ロジック ---

JUDGE_CRITERIA = ["実用性", "共感性", "専門性"]

JUDGE_PROMPT_TEMPLATE = """\
あなたはマネジメントの専門家です。
以下の質問に対する回答を厳密に評価し、必ずJSON形式のみで返してください。

## 質問
{original_prompt}

## 評価対象の回答
{response_text}

## 評価基準 (各軸 1〜5 点)
- 実用性: 具体的なアクションや手順が含まれているか
- 共感性: 相談者の感情を受け止め、心理的安全性を確保しているか
- 専門性: マネジメント理論や業界知識が反映されているか

## 出力形式
以下のJSONのみを返してください。他のテキストは一切含めないでください。
{{"実用性": <1-5>, "共感性": <1-5>, "専門性": <1-5>, "コメント": "<50字以内の総評>"}}
"""

def call_judge(response_text, original_prompt):
    """ベースモデルを審判として呼び出し、スコアを返す"""
    if not VERTEX_API_KEY or not PROJECT_ID:
        return None, "VERTEX_API_KEY または GCP_PROJECT_ID が未設定です。"

    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
        original_prompt=original_prompt,
        response_text=response_text,
    )
    base_url = (
        f"https://{REGION}-aiplatform.googleapis.com/v1beta1"
        f"/projects/{PROJECT_ID}/locations/{REGION}"
        f"/publishers/google/models/{BASE_MODEL_ID}"
    )
    url = f"{base_url}:streamGenerateContent?key={VERTEX_API_KEY}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": judge_prompt}]}],
        "generationConfig": {"maxOutputTokens": 512, "temperature": 0.1},
    }
    headers = {"Content-Type": "application/json"}
    data = json.dumps(payload).encode("utf-8")

    full_text = ""
    try:
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req) as response:
            resp_list = json.loads(response.read().decode("utf-8"))
            for item in resp_list:
                for cand in item.get("candidates", []):
                    for part in cand.get("content", {}).get("parts", []):
                        full_text += part.get("text", "")
    except Exception as e:
        return None, str(e)

    # JSONを抽出
    match = re.search(r"\{.*\}", full_text, re.DOTALL)
    if not match:
        return None, f"JSONが取得できませんでした: {full_text[:200]}"
    try:
        result = json.loads(match.group())
        return result, None
    except json.JSONDecodeError as e:
        return None, f"JSONパースエラー: {e} / raw: {match.group()[:200]}"

def format_judge_report(base_judgment, tuned_judgment, base_error, tuned_error):
    """審判スコアを比較してMarkdown形式のレポートを返す"""
    report = "## 🧑\u200d⚖️ AI審判による評価\n\n"

    def render_table(judgment, error):
        if error or judgment is None:
            return f"> 評価の取得に失敗しました: {error}\n"
        lines = [
            "| 軸 | スコア |",
            "|---|---|",
        ]
        total = 0
        for key in JUDGE_CRITERIA:
            score = judgment.get(key, "-")
            if isinstance(score, int):
                total += score
            lines.append(f"| {key} | {score}/5 |")
        lines.append(f"| **合計** | **{total}/15** |")
        comment = judgment.get("コメント", "")
        return "\n".join(lines) + (f"\n\n> {comment}" if comment else "")

    report += "### 🔹 ベースモデル\n"
    report += render_table(base_judgment, base_error) + "\n\n"
    report += "### 🔸 チューニング済みモデル\n"
    report += render_table(tuned_judgment, tuned_error) + "\n\n"

    # 判定
    if base_judgment and tuned_judgment:
        base_total = sum(base_judgment.get(k, 0) for k in JUDGE_CRITERIA)
        tuned_total = sum(tuned_judgment.get(k, 0) for k in JUDGE_CRITERIA)
        diff = tuned_total - base_total
        sign = "+" if diff >= 0 else ""
        if diff > 0:
            verdict = f"**判定:** チューニング済みモデルが {sign}{diff}点 上回っています。 ✅"
        elif diff == 0:
            verdict = "**判定:** 両モデルは同点です。 ➖"
        else:
            verdict = f"**判定:** ベースモデルが {abs(diff)}点 上回っています。 ⚠️"
        report += verdict + "\n"

    return report

# --- 意味的類似度 (埋め込みベクトル) ロジック ---

EMBEDDING_MODEL_ID = "text-multilingual-embedding-002"

def get_embedding(text):
    """テキストをVertex AI埋め込みモデルでベクトル化して返す"""
    if not VERTEX_API_KEY or not PROJECT_ID:
        raise RuntimeError("VERTEX_API_KEY または GCP_PROJECT_ID が未設定です。")

    url = (
        f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}"
        f"/locations/{REGION}/publishers/google/models/{EMBEDDING_MODEL_ID}:predict"
        f"?key={VERTEX_API_KEY}"
    )
    payload = {"instances": [{"content": text[:3000]}]}  # API上限対応
    headers = {"Content-Type": "application/json"}
    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode("utf-8"))
    return result["predictions"][0]["embeddings"]["values"]

def cosine_similarity(vec1, vec2):
    """コサイン類似度を計算して返す (0、1の範囲)"""
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

def format_similarity_report(base_sim, tuned_sim):
    """類似度スコアを比較してMarkdown形式のレポートを返す"""
    diff = tuned_sim - base_sim
    sign = "+" if diff >= 0 else ""
    mark = "✅" if diff > 0 else ("➖" if diff == 0 else "⚠️")

    table_lines = [
        "| モデル | 類似度スコア |",
        "|---|---|",
        f"| 🔹 ベースモデル | {base_sim:.4f} |",
        f"| 🔸 チューニング済み | {tuned_sim:.4f} |",
        f"| **差分** | **{sign}{diff:.4f} {mark}** |",
    ]

    if diff > 0:
        verdict = "チューニング済みモデルの回答が参照回答により近い意味を持っています。"
    elif diff == 0:
        verdict = "両モデルの類似度は同等です。"
    else:
        verdict = "ベースモデルの回答が参照回答により近い意味を持っています。チューニングデータの見直しを検討してください。"

    report = "## 🔍 意味的類似度評価 (BERTScore-like)\n\n"
    report += "参照回答との意味的な近さを Vertex AI 埋め込みモデルで計測しました。\n\n"
    report += "\n".join(table_lines)
    report += f"\n\n**判定:** {verdict}\n"
    return report
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run specific model comparison tasks.")
    parser.add_argument("prompt", nargs="?", help="The prompt to send to the models.")
    parser.add_argument("--mode", choices=["parse", "parse-reference", "base", "tuned", "simultaneous", "evaluate", "judge", "similarity"], default="simultaneous", help="Execution mode.")
    parser.add_argument("--body", help="Issue body content for parsing prompt.")
    parser.add_argument("--base-file", help="Base model result file path (for evaluate/judge/similarity mode).")
    parser.add_argument("--tuned-file", help="Tuned model result file path (for evaluate/judge/similarity mode).")
    parser.add_argument("--prompt-text", help="Original prompt text (for judge mode).")
    parser.add_argument("--reference-text", help="Reference answer text (for similarity mode).")
    parser.add_argument("--output", default="score_result.md", help="Output file path for evaluate/judge/similarity mode.")
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

    # --- Mode: Similarity (意味的類似度) ---
    if args.mode == "similarity":
        base_file = args.base_file
        tuned_file = args.tuned_file
        reference_text = args.reference_text
        if not base_file or not tuned_file or not reference_text:
            print("Error: --base-file、--tuned-file、--reference-text が必要です。")
            sys.exit(1)
        if not VERTEX_API_KEY or not PROJECT_ID:
            print("Error: VERTEX_API_KEY または GCP_PROJECT_ID が未設定です。")
            sys.exit(1)
        with open(base_file, "r", encoding="utf-8") as f:
            base_text = f.read()
        with open(tuned_file, "r", encoding="utf-8") as f:
            tuned_text = f.read()
        print("🔍 埋め込みベクトルを取得中...")
        ref_vec = get_embedding(reference_text)
        base_vec = get_embedding(base_text)
        tuned_vec = get_embedding(tuned_text)
        base_sim = cosine_similarity(ref_vec, base_vec)
        tuned_sim = cosine_similarity(ref_vec, tuned_vec)
        report = format_similarity_report(base_sim, tuned_sim)
        output_path = args.output
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(report)
        return

    # --- Mode: Judge (LLM-as-Judge) ---
    if args.mode == "judge":
        base_file = args.base_file
        tuned_file = args.tuned_file
        original_prompt = args.prompt_text or DEFAULT_PROMPT
        if not base_file or not tuned_file:
            print("Error: --base-file と --tuned-file が必要です。")
            sys.exit(1)
        with open(base_file, "r", encoding="utf-8") as f:
            base_text = f.read()
        with open(tuned_file, "r", encoding="utf-8") as f:
            tuned_text = f.read()
        print("🧑‍⚖️ ベースモデルの回答を採点中...")
        base_judgment, base_error = call_judge(base_text, original_prompt)
        print("🧑‍⚖️ チューニング済みモデルの回答を採点中...")
        tuned_judgment, tuned_error = call_judge(tuned_text, original_prompt)
        report = format_judge_report(base_judgment, tuned_judgment, base_error, tuned_error)
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
        match = re.search(r"### Prompt\s*\n\s*(.*)", content, re.DOTALL)
        if match:
            print(match.group(1).strip())
        else:
            print(content.strip())
        return

    # --- Mode: Parse Reference Answer from Issue Body ---
    if args.mode == "parse-reference":
        content = args.body or os.environ.get("ISSUE_BODY", "")
        match = re.search(r"### 参照回答\s*\n\s*(.*)", content, re.DOTALL)
        if match:
            text = match.group(1).strip()
            # 次のセクション (### で始まる行) があれば手前で切る
            text = re.split(r"^###", text, maxsplit=1, flags=re.MULTILINE)[0].strip()
            print(text)
        # 参照回答がない場合は何も返さない (スキップのトリガーになる)
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

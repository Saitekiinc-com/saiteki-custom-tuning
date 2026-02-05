import streamlit as st
import json
import urllib.request
import urllib.error
import time

# --- Configuration ---
PROJECT_ID = "gen-lang-client-0646195883"
REGION = "us-central1"
# Tuned Endpoint
TUNED_ENDPOINT_ID = "3814882036406026240"
# Base Model ID
BASE_MODEL_ID = "gemini-2.0-flash-001"

# In a real deployed app, use secrets or environment variables!
API_KEY = "AQ.Ab8RN6Kx86NMYSqVQfPivmZ26P7MjS3i6THSEyjKP7fEFmlHfA"

st.set_page_config(page_title="Gemini Model Comparison", page_icon="⚖️", layout="wide")

st.title("⚖️ Gemini Comparison: Base vs Tuned")
st.markdown("一つのプロンプトを両方のモデルに送信し、回答の違いを比較します。")

# --- API Functions ---

def get_base_model_response(prompt):
    url = f"https://{REGION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT_ID}/locations/{REGION}/publishers/google/models/{BASE_MODEL_ID}:streamGenerateContent?key={API_KEY}"
    return call_api(url, prompt)

def get_tuned_model_response(prompt):
    url = f"https://{REGION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/{TUNED_ENDPOINT_ID}:streamGenerateContent?key={API_KEY}"
    return call_api(url, prompt)

def call_api(url, prompt):
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 800, "temperature": 0.7}
    }
    
    headers = {"Content-Type": "application/json"}
    data = json.dumps(payload).encode("utf-8")
    
    full_text = ""
    try:
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req) as response:
            while True:
                line = response.readline()
                if not line: break
                try:
                    line_str = line.decode("utf-8")
                    if '"text":' in line_str:
                        import re
                        match = re.search(r'"text":\s*"(.*)"', line_str)
                        if match:
                           chunk = match.group(1).encode('utf-8').decode('unicode_escape')
                           full_text += chunk
                except:
                    pass
    except Exception as e:
        return f"Error: {e}"
    return full_text

# --- UI Layout ---

# Input Area
with st.container():
    prompt = st.text_area("プロンプトを入力してください", height=100, placeholder="例: 部下から日報が提出されましたが内容が薄いです。どう指摘しますか？")
    submit_btn = st.button("送信して比較する", type="primary")

if submit_btn and prompt:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔹 Base Model (Untuned)")
        st.caption(f"Model: {BASE_MODEL_ID}")
        with st.spinner("Base Model 生成中..."):
            base_res = get_base_model_response(prompt)
            st.info(base_res)

    with col2:
        st.subheader("🔸 Tuned Model (Fine-Tuned)")
        st.caption(f"Endpoint: {TUNED_ENDPOINT_ID}")
        with st.spinner("Tuned Model 生成中..."):
            tuned_res = get_tuned_model_response(prompt)
            st.success(tuned_res)

    st.divider()
    st.markdown("### 💡 比較のポイント")
    st.markdown("- **Base Model**: 一般的、教科書的、抽象度が高い回答になりがちです。")
    st.markdown("- **Tuned Model**: 独自のルール、役割（ペルソナ）、具体的なフォーマットに従った回答が期待できます。")

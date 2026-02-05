import streamlit as st
import json
import urllib.request
import urllib.error
import os

# --- Configuration ---
PROJECT_ID = "gen-lang-client-0646195883"
REGION = "us-central1"
ENDPOINT_ID = "3814882036406026240"
# In a real deployed app, use secrets or environment variables for the API Key!
API_KEY = "AQ.Ab8RN6Kx86NMYSqVQfPivmZ26P7MjS3i6THSEyjKP7fEFmlHfA"

st.set_page_config(page_title="Saiteki Tuned Model Chat", page_icon="🤖")

st.title("🤖 Saiteki Custom Gemini Chat")
st.caption(f"Tuned Model Endpoint: {ENDPOINT_ID}")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to call Vertex AI API
def get_model_response(messages):
    url = f"https://{REGION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}:streamGenerateContent?key={API_KEY}"
    
    # Convert Streamlit messages to Gemini API format
    gemini_contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        gemini_contents.append({
            "role": role,
            "parts": [{"text": msg["content"]}]
        })
    
    payload = {
        "contents": gemini_contents,
        "generationConfig": {"maxOutputTokens": 500}
    }
    
    headers = {"Content-Type": "application/json"}
    data = json.dumps(payload).encode("utf-8")
    
    full_text = ""
    try:
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req) as response:
            # Simple handling of the response stream
            while True:
                line = response.readline()
                if not line: break
                try:
                    # Crude parsing for the text parts in the stream
                    # A robust implementation would use a proper SSE parser or the Google Cloud SDK
                    line_str = line.decode("utf-8")
                    if '"text":' in line_str:
                        import re
                        match = re.search(r'"text":\s*"(.*)"', line_str)
                        if match:
                           # JSON string unescaping needed
                           chunk = match.group(1).encode('utf-8').decode('unicode_escape')
                           full_text += chunk
                except:
                    pass
    except Exception as e:
        return f"Error: {e}"
        
    return full_text

# React to user input
if prompt := st.chat_input("質問を入力してください..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("モデルが思考中..."):
            response = get_model_response(st.session_state.messages)
            st.markdown(response)
            
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

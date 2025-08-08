# app/main.py
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from core.file_handler import save_uploaded_file, read_uploaded_file

st.set_page_config(page_title="Stat Explainer", layout="wide")

st.title("📤 上傳模型或資料檔案")
uploaded_file = st.file_uploader(
    "請上傳模型檔案（csv / json / pkl）", type=["csv", "json", "pkl"]
)


from core.model_explainer import explain_model

# 在檔案預覽下方
if uploaded_file:
    file_path = save_uploaded_file(uploaded_file)
    st.success(f"✅ 已儲存至: {file_path}")
    preview = read_uploaded_file(file_path)
    st.subheader("📋 檔案預覽")
    st.dataframe(preview)

    # 加入 LLM 解釋功能
    st.subheader("🧠 使用 GPT 解釋模型")
    if st.button("📖 解釋這份模型內容"):
        with st.spinner("LLM 分析中，請稍候..."):
            result = explain_model(
                preview.to_csv(index=False), file_type=uploaded_file.type
            )
        st.text_area("🔍 GPT 解釋結果", value=result, height=300)

import os
from core.rag_chain import run_rag_pipeline
st.subheader("📎 上傳背景說明檔案（.txt）")
rag_file = st.file_uploader("選擇背景說明檔（純文字）", type=["txt"], key="rag")

rag_text = None
if rag_file:
    rag_bytes = rag_file.read()
    rag_text = rag_bytes.decode("utf-8")
    st.text_area("📝 檢視背景內容", value=rag_text, height=200)

    if uploaded_file:
        question = "請根據這份背景說明來解釋上傳的模型結果。"
        if st.button("📖 使用 RAG 解釋模型"):
            with st.spinner("正在檢索與分析..."):
                response = run_rag_pipeline(question=question, raw_text=rag_text)
            st.text_area("🔍 GPT（RAG）回應", value=response, height=300)
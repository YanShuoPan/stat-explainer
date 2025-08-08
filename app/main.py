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

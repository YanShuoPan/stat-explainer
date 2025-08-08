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

if uploaded_file:
    file_path = save_uploaded_file(uploaded_file)
    st.success(f"✅ 已儲存至: {file_path}")

    # 嘗試預覽內容
    preview = read_uploaded_file(file_path)
    st.subheader("📋 檔案預覽")
    st.dataframe(preview)

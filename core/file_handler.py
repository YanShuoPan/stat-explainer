# core/file_handler.py

import io
import os

import pandas as pd

UPLOAD_DIR = "data/uploads"


def ensure_upload_dir():
    """確保上傳資料夾存在"""
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)


def save_uploaded_file(uploaded_file):
    """
    儲存上傳的檔案到 UPLOAD_DIR
    - uploaded_file: Streamlit file_uploader 回傳的檔案物件
    """
    ensure_upload_dir()
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def read_uploaded_file(uploaded_file):
    """
    讀取上傳的檔案（目前支援 CSV & TXT）
    回傳 DataFrame 或文字
    """
    if uploaded_file is None:
        return None

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    elif file_name.endswith(".txt"):
        # 讀成純文字
        return uploaded_file.getvalue().decode("utf-8")

    else:
        raise ValueError("目前僅支援 CSV 和 TXT 檔案。")

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


def read_uploaded_file(file_input):
    """
    讀取檔案，可接受：
    - Streamlit UploadedFile
    - 檔案路徑 (str)
    """
    if file_input is None:
        return None

    # 如果是 UploadedFile
    if hasattr(file_input, "name"):
        file_name = file_input.name.lower()
        buffer = file_input
    else:
        # 假設是字串路徑
        file_name = str(file_input).lower()
        buffer = open(file_input, "rb")

    if file_name.endswith(".csv"):
        return pd.read_csv(buffer)

    elif file_name.endswith(".txt"):
        return buffer.read().decode("utf-8")

    else:
        raise ValueError("目前僅支援 CSV 和 TXT 檔案。")

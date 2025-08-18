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

    if file_input.endswith(".csv"):
        return pd.read_csv(file_input)

    elif file_input.endswith(".txt"):
        return file_input.read().decode("utf-8")

    else:
        raise ValueError("目前僅支援 CSV 和 TXT 檔案。")

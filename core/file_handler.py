# core/file_handler.py
import os

import pandas as pd

UPLOAD_DIR = "data/uploads"


def ensure_upload_dir():
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)


def save_uploaded_file(uploaded_file):
    """儲存上傳的檔案到 UPLOAD_DIR"""
    ensure_upload_dir()
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def read_uploaded_file(file_input):
    """
    讀取檔案
    - 若輸入是 Streamlit UploadedFile → 直接讀
    - 若輸入是 str (檔案路徑) → 開檔讀
    """
    if file_input is None:
        return None

    if hasattr(file_input, "name"):  # UploadedFile
        file_name = file_input.name.lower()
        buffer = file_input
    else:  # str path
        file_name = str(file_input).lower()
        buffer = open(file_input, "rb")

    if file_name.endswith(".csv"):
        return pd.read_csv(buffer)
    elif file_name.endswith(".txt"):
        return buffer.read().decode("utf-8")
    else:
        raise ValueError("目前僅支援 CSV 和 TXT 檔案。")

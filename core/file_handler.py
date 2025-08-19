# core/file_handler.py

import json
import os
import pickle

import pandas as pd

UPLOAD_DIR = "data"


def save_uploaded_file(uploaded_file):
    """接受 Streamlit UploadedFile 或 str 路徑，存到 data/ 下。"""
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)

    # Case 1: Streamlit UploadedFile
    if hasattr(uploaded_file, "name") and hasattr(uploaded_file, "getbuffer"):
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path

    # Case 2: 傳進來本地路徑字串
    elif isinstance(uploaded_file, str):
        # 直接 copy 到 data 資料夾
        file_name = os.path.basename(uploaded_file)
        file_path = os.path.join(UPLOAD_DIR, file_name)
        with open(uploaded_file, "rb") as src, open(file_path, "wb") as dst:
            dst.write(src.read())
        return file_path

    else:
        raise TypeError("save_uploaded_file 只支援 UploadedFile 或 str 檔案路徑")


def read_uploaded_file(file_path):
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith(".json"):
        with open(file_path, "r") as f:
            return pd.DataFrame(json.load(f))
    elif file_path.endswith(".pkl"):
        with open(file_path, "rb") as f:
            obj = pickle.load(f)
            if isinstance(obj, pd.DataFrame):
                return obj
            else:
                return pd.DataFrame({"info": [str(obj)]})
    else:
        return pd.DataFrame({"Error": ["不支援的檔案格式"]})

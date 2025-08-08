# core/file_handler.py

import os
import pandas as pd
import pickle
import json

UPLOAD_DIR = "data"

def save_uploaded_file(uploaded_file):
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)

    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


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

# core/file_handler.py
import io
import os

import pandas as pd

UPLOAD_DIR = "data/uploads"


def ensure_upload_dir():
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR, exist_ok=True)


def save_uploaded_file(uploaded_file):
    """
    把 Streamlit 的 UploadedFile 存成檔案，並回傳存檔路徑。
    """
    ensure_upload_dir()
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def read_uploaded_file(file_input):
    """
    可同時接受：
    1. Streamlit UploadedFile (有 .name 屬性)
    2. 檔案路徑字串
    支援 CSV / JSON / PKL / TXT
    """
    # case 1: UploadedFile
    if hasattr(file_input, "name"):
        name_lower = file_input.name.lower()
        if name_lower.endswith(".csv"):
            return pd.read_csv(file_input)
        if name_lower.endswith(".json"):
            return pd.read_json(file_input)
        if name_lower.endswith(".pkl") or name_lower.endswith(".pickle"):
            data = file_input.getvalue()
            return pd.read_pickle(io.BytesIO(data))
        if name_lower.endswith(".txt"):
            return file_input.getvalue().decode("utf-8", errors="ignore")
        raise ValueError(f"不支援的檔案格式: {file_input.name}")

    # case 2: 字串路徑
    if isinstance(file_input, str):
        name_lower = file_input.lower()
        if name_lower.endswith(".csv"):
            return pd.read_csv(file_input)
        if name_lower.endswith(".json"):
            return pd.read_json(file_input)
        if name_lower.endswith(".pkl") or name_lower.endswith(".pickle"):
            return pd.read_pickle(file_input)
        if name_lower.endswith(".txt"):
            with open(file_input, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        raise ValueError(f"不支援的檔案格式: {file_input}")

    raise TypeError("read_uploaded_file 只接受 UploadedFile 或 str 路徑")

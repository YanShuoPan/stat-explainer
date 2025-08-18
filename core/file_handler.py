# core/file_handler.py
from __future__ import annotations

import io
import os
from typing import Any

import pandas as pd

UPLOAD_DIR = "data/uploads"


def ensure_upload_dir() -> None:
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR, exist_ok=True)


def save_uploaded_file(uploaded_file: Any) -> str:
    """
    將 Streamlit 的 UploadedFile 儲存到 data/uploads/ 並回傳完整路徑。
    """
    ensure_upload_dir()
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def read_uploaded_file(file_input: Any):
    """
    讀取檔案內容，可接受 2 種型別：
    - Streamlit UploadedFile 物件（有 .name）
    - 檔案路徑字串 str（本地磁碟路徑）

    支援：.csv、.txt、.json、.pkl
    csv/json 以 pandas DataFrame 回傳；txt 回傳字串；pkl 回傳原物件/DF。
    """
    if file_input is None:
        return None

    # 1) UploadedFile 物件
    if hasattr(file_input, "name"):
        name_lower = file_input.name.lower()
        if name_lower.endswith(".csv"):
            return pd.read_csv(file_input)
        if name_lower.endswith(".json"):
            return pd.read_json(file_input)
        if name_lower.endswith(".pkl") or name_lower.endswith(".pickle"):
            # 先讀成 bytes 再交給 pandas
            data = file_input.getvalue()
            return pd.read_pickle(io.BytesIO(data))
        if name_lower.endswith(".txt"):
            return file_input.getvalue().decode("utf-8", errors="ignore")
        raise ValueError("目前僅支援 CSV/JSON/PKL/TXT 檔案。")

    # 2) 檔案路徑字串
    path = str(file_input)
    name_lower = path.lower()
    if name_lower.endswith(".csv"):
        return pd.read_csv(path)
    if name_lower.endswith(".json"):
        return pd.read_json(path)
    if name_lower.endswith(".pkl") or name_lower.endswith(".pickle"):
        return pd.read_pickle(path)
    if name_lower.endswith(".txt"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    raise ValueError("目前僅支援 CSV/JSON/PKL/TXT 檔案。")

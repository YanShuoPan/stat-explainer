# core/llm_tools.py
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# 依你的實際位置調整：
from core.ohit import oga_hdic  # 若檔案仍在根目錄，就寫: from Ohit import oga_hdic
from core.tool_registry import register_tool


def _to_jsonable(obj: Any):
    """把 numpy/pandas 轉成可 JSON 的型態，避免序列化失敗。"""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return {"columns": obj.columns.tolist(), "data": obj.to_dict(orient="records")}
    if isinstance(obj, pd.Series):
        return obj.tolist()
    return obj


@register_tool(
    name="run_oga_hdic",
    description="使用 OGA-HDiC 演算法在提供的資料上進行模型選擇／估計。",
    schema={
        "type": "object",
        "properties": {
            "data": {
                "type": "array",
                "items": {"type": "object"},
                "description": "資料列，每列是一個欄位到值的映射。",
            },
            "y_col": {"type": "string", "description": "目標變數的欄位名"},
            "x_cols": {
                "type": "array",
                "items": {"type": "string"},
                "description": "特徵欄位清單；若缺省則使用 y 以外的全部欄位。",
            },
            "Kn": {
                "type": ["integer", "null"],
                "description": "選擇的變數數上限，null 代表不指定",
            },
            "c1": {"type": "number", "default": 5},
            "HDIC_Type": {"type": "string", "default": "HDBIC"},
            "c2": {"type": "number", "default": 2},
            "c3": {"type": "number", "default": 2.01},
            "intercept": {"type": "boolean", "default": True},
        },
        "required": ["data", "y_col"],
    },
)
def run_oga_hdic_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    # 讀入參數
    rows = args["data"]
    y_col = args["y_col"]
    x_cols = args.get("x_cols")
    Kn = args.get("Kn", None)
    c1 = args.get("c1", 5)
    HDIC_Type = args.get("HDIC_Type", "HDBIC")
    c2 = args.get("c2", 2)
    c3 = args.get("c3", 2.01)
    intercept = args.get("intercept", True)

    df = pd.DataFrame(rows)
    if y_col not in df.columns:
        raise ValueError(f"y_col '{y_col}' 不在資料欄位中：{df.columns.tolist()}")

    if not x_cols:
        x_cols = [c for c in df.columns if c != y_col]

    X = df[x_cols].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")

    # 你可以在這裡套用你的缺失值策略（例如丟棄或補值）
    valid = X.notna().all(axis=1) & y.notna()
    X_clean = X.loc[valid]
    y_clean = y.loc[valid]

    # 呼叫你自定義的模型
    result = oga_hdic(
        X_clean.values,
        y_clean.values,
        Kn=Kn,
        c1=c1,
        HDIC_Type=HDIC_Type,
        c2=c2,
        c3=c3,
        intercept=intercept,
    )

    # 嘗試把回傳轉成可序列化
    try:
        if isinstance(result, dict):
            result_json = {k: _to_jsonable(v) for k, v in result.items()}
        else:
            result_json = _to_jsonable(result)
    except Exception:
        result_json = str(result)

    return {
        "used_rows": int(valid.sum()),
        "x_cols": x_cols,
        "y_col": y_col,
        "params": {
            "Kn": Kn,
            "c1": c1,
            "HDIC_Type": HDIC_Type,
            "c2": c2,
            "c3": c3,
            "intercept": intercept,
        },
        "result": result_json,
    }

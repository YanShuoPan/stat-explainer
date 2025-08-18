# core/llm_tools.py
from typing import List
import pandas as pd
from core.tool_registry import register_tool

@register_tool(
    name="run_linear_model_predict",
    description="用線性模型係數對輸入資料做預測，回傳摘要統計與前幾筆預測。",
    schema={
        "type": "object",
        "properties": {
            "coefficients": {
                "type": "object",
                "description": "鍵是欄位名（含 Intercept 選用），值是係數（float）。"
            },
            "data": {
                "type": "array",
                "items": {"type": "object"},
                "description": "每筆是 {欄位: 值} 的資料列。"
            },
            "head": {"type": "integer", "default": 5}
        },
        "required": ["coefficients", "data"]
    }
)
def run_linear_model_predict(args: dict):
    coefs = args["coefficients"]
    rows = args["data"]
    head_n = args.get("head", 5)

    df = pd.DataFrame(rows)
    intercept = float(coefs.get("Intercept", 0.0))
    yhat = intercept
    for col, w in coefs.items():
        if col == "Intercept": 
            continue
        if col in df.columns:
            yhat = yhat + df[col].astype(float) * float(w)
    df["_pred"] = yhat

    summary = {
        "count": int(df["_pred"].count()),
        "mean": float(df["_pred"].mean()),
        "std": float(df["_pred"].std() or 0.0),
        "min": float(df["_pred"].min()),
        "max": float(df["_pred"].max()),
    }
    preview = df.head(head_n).to_dict(orient="records")
    return {"summary": summary, "preview": preview}
# app/main.py — stat-explainer (Trimmed: Upload/Preview + RAG + OGA-HDiC, Level 3)
# -------------------------------------------------------------
# 功能：
# 1) 上傳與預覽 Upload & Preview（csv/json/pkl）
# 2) RAG 增強：上傳 .txt 建索引 → 輔助 GPT 解釋
# 3) 自訂函式：OGA-HDiC（選 y 欄 + 參數）Level 3 模式
#   - 僅本地執行函式，不將原始資料傳給 LLM
#   - 若需要 LLM，只會傳送精簡後結果摘要
# -------------------------------------------------------------

import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import streamlit as st

# 檔案處理
from core.file_handler import read_uploaded_file, save_uploaded_file  # type: ignore
from core.llm_tools import *

# 確保可以 import core/


# RAG 管線
try:
    from core.rag_chain import run_rag_pipeline  # type: ignore
except Exception:  # noqa: BLE001
    run_rag_pipeline = None

# 工具呼叫
from core.llm_executor import make_client  # type: ignore
from core.tool_registry import dispatch_tool  # type: ignore


# -------------------------------------------------------------
# 基本設定
# -------------------------------------------------------------
def to_jsonable(obj, max_list=100, max_str=4000):
    """把 numpy/pandas/statsmodels 結構轉成可 JSON 序列化的純 Python。
    會對長序列/字串做截斷，避免 payload 過大。"""
    # 基本型別
    if obj is None or isinstance(obj, (bool, int, float, str)):
        if isinstance(obj, str) and len(obj) > max_str:
            return obj[:max_str] + " ...<truncated>"
        return obj

    # numpy 標量
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()

    # pandas
    if isinstance(obj, pd.Series):
        lst = obj.tolist()
        if len(lst) > max_list:
            return lst[:max_list] + ["<truncated>"]
        return lst
    if isinstance(obj, pd.Index):
        lst = obj.tolist()
        if len(lst) > max_list:
            return lst[:max_list] + ["<truncated>"]
        return lst
    if isinstance(obj, pd.DataFrame):
        # 只輸出前 max_list 列
        df_small = obj.head(max_list)
        return {
            "columns": df_small.columns.tolist(),
            "rows": df_small.to_dict(orient="records"),
            "note": f"truncated to {len(df_small)} rows" if len(obj) > max_list else "full",
        }

    # numpy 陣列/序列
    if isinstance(obj, (list, tuple, set)):
        lst = list(obj)
        if len(lst) > max_list:
            lst = lst[:max_list] + ["<truncated>"]
        return [to_jsonable(v, max_list=max_list, max_str=max_str) for v in lst]
    if isinstance(obj, np.ndarray):
        lst = obj.tolist()
        if len(lst) > max_list:
            lst = lst[:max_list] + ["<truncated>"]
        return lst

    # dict
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out[str(k)] = to_jsonable(v, max_list=max_list, max_str=max_str)
        return out

    # statsmodels 等其他複雜物件 → 嘗試抓常用屬性，否則轉字串
    for attr in ("params", "pvalues", "rsquared", "aic", "bic", "bic_llf", "llf"):
        if hasattr(obj, attr):
            try:
                return to_jsonable(getattr(obj, attr), max_list=max_list, max_str=max_str)
            except Exception:
                pass

    return str(obj)[:max_str] + (" ...<truncated>" if len(str(obj)) > max_str else "")


st.set_page_config(page_title="Stat Explainer", layout="wide")
st.title("📊 stat-explainer — 上傳/預覽 + RAG + OGA-HDiC (Level 3)")


# 取得 API Key
def get_api_key() -> str | None:
    return st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))


API_KEY = get_api_key()
if not API_KEY:
    st.warning("未找到 OPENAI_API_KEY（Secrets 或環境變數），RAG 功能將無法呼叫 GPT。")

# =============================================================
# 1) 上傳與預覽
# =============================================================
st.header("1) 上傳與預覽 Upload & Preview")

uploaded_file = st.file_uploader(
    "請上傳資料/模型輸出檔（csv / json / pkl）",
    type=["csv", "json", "pkl"],
    key="data_file",
)
preview: pd.DataFrame | None = None
file_path: str | None = None

if uploaded_file:
    file_path = save_uploaded_file(uploaded_file)
    st.success(f"✅ 已儲存至: {file_path}")
    preview = read_uploaded_file(file_path)
    st.subheader("📋 檔案預覽")
    if isinstance(preview, pd.DataFrame):
        st.dataframe(preview, use_container_width=True)
        st.caption(f"Rows: {preview.shape[0]} | Cols: {preview.shape[1]}")
    else:
        st.write(preview)

# # =============================================================
# # 2) RAG 增強（.txt 背景）
# # =============================================================
# st.header("2) RAG 增強（上傳 .txt 背景 → 檢索 + 解釋)")

# rag_file = st.file_uploader("上傳背景說明檔（純文字 .txt）", type=["txt"], key="rag_file")
# rag_text: str | None = None
# if rag_file is not None:
#     try:
#         rag_text = rag_file.read().decode("utf-8", errors="ignore")
#     except Exception:  # noqa: BLE001
#         rag_text = None

# if rag_text:
#     st.text_area("📝 背景內容預覽", rag_text[:5000], height=180)

# if st.button("📖 使用 RAG 解釋", disabled=(rag_text is None or preview is None)):
#     if not API_KEY:
#         st.error("缺少 OPENAI_API_KEY，無法執行 RAG。")
#     elif run_rag_pipeline is None:
#         st.error("未發現 core.rag_chain.run_rag_pipeline。")
#     else:
#         try:
#             question = "請根據背景說明與上傳資料，解釋這份模型/資料的重點與結論。"
#             response = run_rag_pipeline(question=question, raw_text=rag_text)  # type: ignore[arg-type]
#             st.text_area("🔍 GPT（RAG）回應", value=response, height=320)
#         except Exception as e:
#             st.error(f"RAG 執行錯誤：{e}")

# =============================================================
# 3) 自訂函式：OGA-HDiC（Level 3 本地 + 可選 LLM 摘要）
# =============================================================
st.header("3) 自訂函式：OGA-HDiC（選擇 y 欄 + 參數）")

if preview is None or not isinstance(preview, pd.DataFrame):
    st.info("請先於上方上傳資料檔案（csv/json/pkl）。")
else:
    df = preview
    cols = list(df.columns)
    if not cols:
        st.warning("資料沒有欄位可供選擇。")
    else:
        y_col = st.selectbox("選擇目標變數 y 欄位", options=cols, index=0)
        default_x = [c for c in cols if c != y_col]
        x_cols = st.multiselect(
            "選擇特徵欄位 X（預設為 y 以外全部）",
            options=cols,
            default=default_x,
        )

        with st.expander("參數設定", expanded=False):
            Kn = st.number_input("Kn（0=不指定）", value=0, min_value=0, step=1)
            c1 = st.number_input("c1", value=5.0, step=0.5)
            HDIC_Type = st.selectbox("HDIC_Type", options=["HDBIC", "HDHQ", "HDAIC"], index=0)
            c2 = st.number_input("c2", value=2.0, step=0.5)
            c3 = st.number_input("c3", value=2.01, step=0.01)
            intercept = st.checkbox("intercept", value=True)

        # ---- 本地執行（不將原始資料送 LLM）----
        st.caption("💡 Level 3：只在本地執行 OGA-HDiC，若要 LLM 摘要，只傳送精簡結果。")

        use_cols = [y_col] + x_cols if x_cols else [y_col]
        sub_df = df[use_cols].copy()
        sub_df = sub_df.apply(pd.to_numeric, errors="ignore")
        data_records_local = sub_df.to_dict(orient="records")

        Kn_arg = None if Kn == 0 else int(Kn)
        tool_args = {
            "data": data_records_local,
            "y_col": y_col,
            "x_cols": x_cols,
            "Kn": Kn_arg,
            "c1": c1,
            "HDIC_Type": HDIC_Type,
            "c2": c2,
            "c3": c3,
            "intercept": intercept,
        }

        col1, col2 = st.columns([1, 1])
        with col1:
            run_local = st.button("⚙️ 僅本地執行 OGA-HDiC")
        with col2:
            run_local_summary = st.button("📝 本地執行 + LLM 摘要")

        if run_local:
            try:
                result = dispatch_tool("run_oga_hdic", tool_args)
                st.success("✅ 本地 OGA-HDiC 完成。")
                with st.expander("🔧 本地結果 JSON"):
                    st.json(result)
            except Exception as e:
                st.error(f"OGA-HDiC 本地執行錯誤：{e}")
        summary_model = st.selectbox(
            "選擇 LLM 模型（摘要用）", ["gpt-4o-mini", "gpt-4o"], index=0, key="oga_summary_model"
        )
        if run_local_summary:
            try:
                result = dispatch_tool("run_oga_hdic", tool_args)
                st.success("✅ 本地 OGA-HDiC 完成，準備請 LLM 摘要…")
                with st.expander("🔧 本地結果 JSON"):
                    st.json(result)

                if not API_KEY:
                    st.warning("未設定 OPENAI_API_KEY，跳過 LLM 摘要。")
                else:
                    client = make_client(API_KEY)

                    # 1) 先過濾大型欄位
                    filtered = dict(result)
                    for k in [
                        "X",
                        "y",
                        "coef_matrix",
                        "residuals",
                        "fitted_values",
                        "influence",
                        "cov_params",
                    ]:
                        if k in filtered:
                            filtered[k] = "<omitted>"

                    # 2) 轉為 JSON-safe（解決 numpy/pandas/ResultWrapper 等型別）
                    safe = to_jsonable(filtered, max_list=100, max_str=4000)

                    # 3) 序列化（separators 去空白，減 token）
                    payload = json.dumps(safe, ensure_ascii=False, separators=(",", ":"))

                    summary = (
                        client.chat.completions.create(
                            model=summary_model,
                            messages=[
                                {
                                    "role": "system",
                                    "content": "你是統計助理，請以繁體中文摘要研究結果。",
                                },
                                {
                                    "role": "user",
                                    "content": "以下是 OGA-HDiC 的結果，'J_Trim'是選到的重要變數，請整理要點：選到的變數、關鍵指標與建議。",
                                },
                                {"role": "user", "content": payload},
                            ],
                            temperature=0.2,
                        )
                        .choices[0]
                        .message.content
                    )

                    st.subheader("📝 LLM 摘要")
                    st.write(summary)

            except Exception as e:
                st.error(f"OGA-HDiC 執行或 LLM 摘要錯誤：{e}")

# -------------------------------------------------------------
# End of file
# -------------------------------------------------------------

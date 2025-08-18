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

import pandas as pd
import streamlit as st

# 檔案處理
from core.file_handler import read_uploaded_file, save_uploaded_file  # type: ignore
from core.llm_tools import dispatch_tool

# 確保可以 import core/


# RAG 管線
try:
    from core.rag_chain import run_rag_pipeline  # type: ignore
except Exception:  # noqa: BLE001
    run_rag_pipeline = None

# 工具呼叫
from core.llm_executor import make_client  # type: ignore
from core.tool_registry import dispatch_tool  # type: ignore

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# -------------------------------------------------------------
# 基本設定
# -------------------------------------------------------------

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

# =============================================================
# 2) RAG 增強（.txt 背景）
# =============================================================
st.header("2) RAG 增強（上傳 .txt 背景 → 檢索 + 解釋)")

rag_file = st.file_uploader("上傳背景說明檔（純文字 .txt）", type=["txt"], key="rag_file")
rag_text: str | None = None
if rag_file is not None:
    try:
        rag_text = rag_file.read().decode("utf-8", errors="ignore")
    except Exception:  # noqa: BLE001
        rag_text = None

if rag_text:
    st.text_area("📝 背景內容預覽", rag_text[:5000], height=180)

if st.button("📖 使用 RAG 解釋", disabled=(rag_text is None or preview is None)):
    if not API_KEY:
        st.error("缺少 OPENAI_API_KEY，無法執行 RAG。")
    elif run_rag_pipeline is None:
        st.error("未發現 core.rag_chain.run_rag_pipeline。")
    else:
        try:
            question = "請根據背景說明與上傳資料，解釋這份模型/資料的重點與結論。"
            response = run_rag_pipeline(question=question, raw_text=rag_text)  # type: ignore[arg-type]
            st.text_area("🔍 GPT（RAG）回應", value=response, height=320)
        except Exception as e:
            st.error(f"RAG 執行錯誤：{e}")

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

        if run_local_summary:
            try:
                result = dispatch_tool("run_oga_hdic", tool_args)
                st.success("✅ 本地 OGA-HDiC 完成，準備請 LLM 摘要…")
                with st.expander("🔧 本地結果 JSON"):
                    st.json(result)
                if API_KEY:
                    client = make_client(API_KEY)
                    compact = dict(result)
                    for k in ["X", "y", "coef_matrix", "residuals", "fitted_values"]:
                        if k in compact:
                            compact[k] = "<omitted>"
                    payload = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))

                    model_name = st.selectbox(
                        "選擇 LLM 模型（摘要用）", ["gpt-4o-mini", "gpt-4o"], index=0
                    )
                    summary = (
                        client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {
                                    "role": "system",
                                    "content": "你是統計助理，請以繁體中文摘要研究結果。",
                                },
                                {
                                    "role": "user",
                                    "content": "以下是 OGA-HDiC 的結果，請整理要點：選到的變數、關鍵指標與建議。",
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
                else:
                    st.warning("未設定 OPENAI_API_KEY，跳過 LLM 摘要。")
            except Exception as e:
                st.error(f"OGA-HDiC 執行或 LLM 摘要錯誤：{e}")

# -------------------------------------------------------------
# End of file
# -------------------------------------------------------------

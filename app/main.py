
import os
import sys
import json
import streamlit as st
import pandas as pd

# Ensure project root on sys.path so `core/*` is importable when running `app/main.py`
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



# --- Core imports (these should exist in your repo) ---
from core.file_handler import save_uploaded_file, read_uploaded_file  # type: ignore

# GPT explainer (non-RAG). If missing, we handle gracefully below.
try:
    from core.model_explainer import explain_model  # type: ignore
except Exception:  # noqa: BLE001
    explain_model = None  # fallback handled later

# RAG pipeline (Chroma + OpenAI embeddings)
try:
    from core.rag_chain import run_rag_pipeline  # type: ignore
except Exception:  # noqa: BLE001
    run_rag_pipeline = None

# Tool-calling executor & registry
from core.llm_executor import make_client, chat_with_tools  # type: ignore
from core.tool_registry import register_tool  # type: ignore

# Import tools so decorators run and register (e.g., my_model, run_oga_hdic)
# - core.llm_tools should define @register_tool functions
try:
    import core.llm_tools  # noqa: F401
except Exception:
    pass

# ---- Page config ----
st.set_page_config(page_title="Stat Explainer", layout="wide")
st.title("📊 stat-explainer — LLM 平台 (RAG + Tools)")

# ---- Helper: obtain API key (Streamlit Cloud secrets first, else env) ----
def get_api_key() -> str | None:
    return st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

API_KEY = get_api_key()
if not API_KEY:
    st.warning(
        "找不到 OPENAI_API_KEY。請在 Streamlit Secrets 或環境變數中設定，以啟用 GPT / RAG / 工具呼叫。"
    )

# =============================================================
# Section 1 — Upload & Preview
# =============================================================
st.header("1) 上傳與預覽 Upload & Preview")

uploaded_file = st.file_uploader("請上傳資料/模型輸出檔（csv / json / pkl）", type=["csv", "json", "pkl"], key="data")
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
# Section 2 — GPT Explanation (non-RAG)
# =============================================================
st.header("2) GPT 解釋（不含 RAG）")
model_choice = st.selectbox("選擇模型（僅用於展示；explain_model 內部可忽略）", ["gpt-4o", "gpt-4o-mini"], index=0)
if st.button("📖 產生解釋", disabled=preview is None):
    if preview is None:
        st.info("請先上傳檔案。")
    elif not API_KEY:
        st.error("缺少 OPENAI_API_KEY，無法呼叫 GPT。")
    else:
        try:
            if explain_model is None:
                # Lightweight fallback: summarize first rows as text
                text_block = preview.head(50).to_csv(index=False) if isinstance(preview, pd.DataFrame) else str(preview)
                st.warning("未發現 core.model_explainer.explain_model，使用簡易顯示為後備方案。")
                st.text_area("暫時輸出", value=text_block, height=300)
            else:
                # explain_model expects textual content
                csv_text = preview.to_csv(index=False) if isinstance(preview, pd.DataFrame) else str(preview)
                out = explain_model(csv_text, file_type="csv")  # type: ignore[arg-type]
                st.text_area("🔍 GPT 解釋結果", value=out, height=320)
        except Exception as e:  # noqa: BLE001
            st.error(f"解釋時發生錯誤：{e}")

# =============================================================
# Section 3 — RAG: upload .txt background + augment
# =============================================================
st.header("3) RAG 增強（上傳 .txt 背景說明 → 檢索 + 解釋)")
rag_file = st.file_uploader("上傳背景說明檔（純文字 .txt）", type=["txt"], key="rag")
rag_text: str | None = None
if rag_file is not None:
    try:
        rag_text = rag_file.read().decode("utf-8", errors="ignore")
    except Exception:  # noqa: BLE001
        rag_text = None

if rag_text:
    st.text_area("📝 背景內容預覽", rag_text[:5000], height=180)

col_rag1, col_rag2 = st.columns([1, 2])
with col_rag1:
    rag_kick = st.button("📖 使用 RAG 解釋模型", disabled=(rag_text is None or preview is None))
with col_rag2:
    st.caption("說明：會建立/載入向量庫（Chroma），用檢索片段輔助 GPT 生成解釋。")

if rag_kick:
    if not API_KEY:
        st.error("缺少 OPENAI_API_KEY，無法執行 RAG。")
    elif preview is None:
        st.info("請先上傳資料檔案。")
    elif run_rag_pipeline is None:
        st.error("未發現 core.rag_chain.run_rag_pipeline，請確認模組已建立。")
    else:
        try:
            # Basic prompt to couple with RAG context
            question = "請根據背景說明與上傳資料，解釋這份模型/資料的重點與結論。"
            response = run_rag_pipeline(question=question, raw_text=rag_text)  # type: ignore[arg-type]
            st.text_area("🔍 GPT（RAG）回應", value=response, height=320)
        except Exception as e:  # noqa: BLE001
            st.error(f"RAG 執行錯誤：{e}")

# =============================================================
# Section 4 — LLM Tool Calling (demo)
# =============================================================
st.header("4) LLM 工具呼叫（Function Calling）測試")

# Simple demo tool is defined in core.llm_tools as `my_model`.
# We'll let LLM decide to call it.
with st.expander("開啟 / 收合：最小工具呼叫測試"):
    x_val = st.number_input("x", value=3.0, step=1.0)
    y_val = st.number_input("y", value=5.0, step=1.0)
    question = st.text_input(
        "你的問題",
        "請用可用的工具計算當 x=3, y=5 時的模型輸出，並簡要說明。",
    )
    model_name = st.selectbox("選擇模型", ["gpt-4o-mini", "gpt-4o"], index=0, key="tool_model")

    if st.button("🚀 執行工具測試"):
        if not API_KEY:
            st.error("缺少 OPENAI_API_KEY。")
        else:
            try:
                client = make_client(API_KEY)
                messages = [
                    {"role": "system", "content": "你是擅長統計建模與數值運算的助理。"},
                    {"role": "user", "content": question},
                    {"role": "user", "content": f"給定 x={x_val}, y={y_val}"},
                    {"role": "user", "content": "若需要計算，請呼叫 my_model 並傳入參數。"},
                ]
                out = chat_with_tools(client, messages, model=model_name, temperature=0.2)
                st.success("LLM 最終回覆：")
                st.write(out["content"])
                with st.expander("🔧 工具呼叫明細（debug）"):
                    st.json(out["tool_results"])    
            except Exception as e:  # noqa: BLE001
                st.error(f"執行錯誤：{e}")

# =============================================================
# Section 5 — OGA-HDiC (custom function) via tool calling
# =============================================================
st.header("5) 自訂函式：OGA-HDiC（選擇 y 欄 + 參數）")

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
        max_rows = st.number_input("送往 LLM 的資料筆數上限", value=1000, min_value=50, step=50)
        data_records = df.iloc[: int(max_rows)].to_dict(orient="records")
        model_name2 = st.selectbox("選擇模型 (tool calling)", ["gpt-4o-mini", "gpt-4o"], index=0, key="oga_model")

        if st.button("🚀 讓 LLM 執行 OGA-HDiC"):
            if not API_KEY:
                st.error("缺少 OPENAI_API_KEY。")
            else:
                try:
                    client = make_client(API_KEY)
                    Kn_arg = None if Kn == 0 else int(Kn)

                    user_goal = (
                        "請在提供的資料上執行 OGA-HDiC 模型選擇，並說明選到的變數與結果重點。"
                        "若需要計算，請呼叫工具 run_oga_hdic，並填入 data/y_col/x_cols/Kn/c1/HDIC_Type/c2/c3/intercept。"
                    )

                    messages = [
                        {"role": "system", "content": "你是熟悉高維模型選擇的統計助理。"},
                        {"role": "user", "content": user_goal},
                        {"role": "user", "content": (
                            f"y_col={y_col}；x_cols={x_cols}；Kn={Kn_arg}；c1={c1}；"
                            f"HDIC_Type={HDIC_Type}；c2={c2}；c3={c3}；intercept={intercept}。"
                        )},
                        {"role": "user", "content": f"資料共有 {len(df)} 列，這裡提供前 {len(data_records)} 列用於計算。"},
                        {"role": "user", "content": json.dumps({"data": data_records}, ensure_ascii=False)},
                    ]

                    out = chat_with_tools(client, messages, model=model_name2, temperature=0.1)
                    st.success("LLM（含工具執行）回覆：")
                    st.write(out["content"])
                    with st.expander("🔧 工具呼叫明細（結果原始 JSON）"):
                        st.json(out["tool_results"])
                except Exception as e:  # noqa: BLE001
                    st.error(f"OGA-HDiC 執行錯誤：{e}")

# -------------------------------------------------------------
# End of file
# -------------------------------------------------------------

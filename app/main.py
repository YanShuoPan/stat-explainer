# app/main.py 內合適位置加入（確保能 import core/*）
import os
import json
import streamlit as st

# 若你需要：把專案根目錄加到 sys.path（你之前已加過可略）
# import sys, os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.tool_registry import register_tool
from core.llm_executor import make_client, chat_with_tools

# ====== 1) 註冊你的工具（示範：最小數學模型） ======
@register_tool(
    name="my_model",
    description="計算最小數學模型：x^2 + y",
    schema={
        "type": "object",
        "properties": {
            "x": {"type": "number", "description": "數字 x"},
            "y": {"type": "number", "description": "數字 y"},
        },
        "required": ["x", "y"],
    },
)
def my_model(args: dict):
    x = float(args["x"])
    y = float(args["y"])
    return {"result": x ** 2 + y}

# ====== 2) UI：快速測試工具呼叫 ======
st.divider()
st.header("🧪 LLM 工具呼叫（Function Calling）模組化測試")

col1, col2 = st.columns(2)
with col1:
    x_val = st.number_input("x", value=3.0, step=1.0)
with col2:
    y_val = st.number_input("y", value=5.0, step=1.0)

question = st.text_input(
    "你的問題",
    "請用可用的工具計算當 x=3, y=5 時的模型輸出，並簡要說明。"
)

model_name = st.selectbox("選擇模型", ["gpt-4o-mini", "gpt-4o"], index=0)

if st.button("🚀 執行工具測試"):
    # 讀取 API Key：Cloud 用 st.secrets，本地用 .env / 環境變數
    api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    if not api_key:
        st.error("找不到 OPENAI_API_KEY，請在 Streamlit Secrets 或環境變數中設定。")
    else:
        try:
            client = make_client(api_key)
            messages = [
                {"role": "system", "content": "你是擅長統計建模與數值運算的助理。"},
                {"role": "user", "content": question},
                # 把數值也提供給 LLM，讓它更容易決定工具參數
                {"role": "user", "content": f"給定 x={x_val}, y={y_val}"},
                {"role": "user", "content": "若需要計算，請呼叫 my_model 並傳入參數。"},
            ]
            out = chat_with_tools(client, messages, model=model_name, temperature=0.2)

            st.success("LLM 最終回覆：")
            st.write(out["content"])
            with st.expander("🔧 工具呼叫明細（debug）"):
                st.json(out["tool_results"])
        except Exception as e:
            st.error(f"執行錯誤：{e}")

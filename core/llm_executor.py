# core/llm_executor.py
import json
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

from core.tool_registry import dispatch_tool, list_tools_for_openai

# 讀取本地 .env；在 Streamlit Cloud 會從 st.secrets 讀（見 app/main.py）
load_dotenv()


def make_client(api_key: str | None = None) -> OpenAI:
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not found (env or provided).")
    return OpenAI(api_key=key)


def chat_with_tools(
    client: OpenAI,
    messages: List[Dict[str, str]],
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """
    發送一次對話，允許 LLM 呼叫已註冊工具；若有工具呼叫則執行並回餵，再取最終回答。
    回傳：{"content": str, "tool_results": list}
    """
    tools = list_tools_for_openai()

    # 第一次：讓 LLM 決定要不要 tool call
    first = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=temperature,
    )
    msg = first.choices[0].message

    # 沒有工具呼叫 → 直接返回模型回答
    if not msg.tool_calls:
        return {"content": msg.content or "", "tool_results": []}

    tool_messages = []
    results = []
    for tc in msg.tool_calls:
        name = tc.function.name
        args = json.loads(tc.function.arguments or "{}")
        result = dispatch_tool(name, args)  # 真正執行你的 Python 函式
        results.append({"name": name, "args": args, "result": result})
        tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "name": name,
                "content": json.dumps(result, ensure_ascii=False),
            }
        )

    # 第二次：把工具結果回餵給 LLM，請它整理最終自然語言輸出
    final = client.chat.completions.create(
        model=model,
        messages=messages
        + [msg]
        + tool_messages
        + [{"role": "user", "content": "請根據工具結果，給我清楚的結論與重點。"}],
        temperature=temperature,
    )
    return {"content": final.choices[0].message.content, "tool_results": results}

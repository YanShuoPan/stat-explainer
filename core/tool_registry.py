# core/tool_registry.py
from typing import Callable, Dict, Any
import json

# 單一工具的結構
class ToolSpec:
    def __init__(self, name: str, description: str, schema: dict, func: Callable[[dict], Any]):
        self.name = name
        self.description = description
        self.schema = schema      # JSON Schema
        self.func = func          # 實際執行的 Python 函式（接收 dict 參數）

_REGISTRY: Dict[str, ToolSpec] = {}

def register_tool(name: str, description: str, schema: dict):
    """
    Decorator：把函式註冊成 LLM 可呼叫的工具（白名單）。
    被裝飾的函式必須 signature 為 fn(args: dict) -> Any
    """
    def _wrap(func: Callable[[dict], Any]):
        _REGISTRY[name] = ToolSpec(name, description, schema, func)
        return func
    return _wrap

def list_tools_for_openai():
    """轉為 OpenAI chat.completions 所需的 tools 格式"""
    tools = []
    for t in _REGISTRY.values():
        tools.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.schema,
            },
        })
    return tools

def dispatch_tool(name: str, arguments: dict):
    """依名稱派發到已註冊工具。這裡是唯一允許執行的入口。"""
    if name not in _REGISTRY:
        raise ValueError(f"Tool {name} not registered")
    tool = _REGISTRY[name]
    # 你可以在這裡加額外驗證（如 jsonschema 驗證 arguments）
    return tool.func(arguments)

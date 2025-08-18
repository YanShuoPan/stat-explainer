# core/rag_chain.py

import os

from dotenv import load_dotenv
from openai import OpenAI

from core.vector_store import create_vectorstore_from_text, query_vectorstore

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def run_rag_pipeline(question: str, raw_text: str):
    # 建立向量庫
    create_vectorstore_from_text(raw_text)

    # 檢索相關內容
    retrieved_docs = query_vectorstore(question)
    context = "\n---\n".join([doc.page_content for doc in retrieved_docs])

    # 組 prompt
    prompt = f"""
你是一位統計專家，請根據以下背景說明與提問回答問題。
背景文件如下：
{context}

問題是：
{question}

請使用繁體中文簡潔回應。
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ 發生錯誤：{str(e)}"

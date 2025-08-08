# core/model_explainer.py

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def explain_model(file_content: str, file_type: str = "csv") -> str:
    prompt = f"""
你是一位統計專家，請閱讀下列{file_type}資料內容，並用清楚的方式幫我解釋這個模型的結構與結果意義。
若資料中包含變數名稱、係數、p 值、重要性等，請幫我指出關鍵發現與推論。

資料內容如下：

請以繁體中文回答。
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "你是一位熟悉統計模型的資料科學家"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ 發生錯誤：{str(e)}"

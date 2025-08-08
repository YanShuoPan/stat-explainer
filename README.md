# 📊 stat-explainer

Stat-Explainer 是一個以 LLM（大型語言模型）為核心的互動平台，專為統計與資料科學研究人員設計。
使用者可上傳模型成果（如 .csv、.pkl、.json 等），平台會透過 GPT-4o 協助自動解釋模型結構與分析結果。

---

## 🔧 功能特色
- 📤 上傳模型或統計結果檔案（.csv / .json / .pkl）
- 📋 資料預覽（DataFrame 顯示）
- 🤖 使用 GPT-4o 解釋模型變數與結果
- ☁️ 一鍵部署於 Streamlit Cloud，無需安裝即可使用

---

## 🌐 線上體驗網址

👉 [https://stat-explainer.streamlit.app](https://stat-explainer-9s6z6p57rxzaeulymj4erz.streamlit.app/)

---

## 🚀 快速開始（本機端）

```bash
# 安裝環境
pip install -r requirements.txt

# 設定 API 金鑰（於 .env 檔中）
OPENAI_API_KEY=your_api_key_here

# 啟動 App
streamlit run app/main.py
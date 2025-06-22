<h1 align="center">🧠 Sentiment-Analysis-driven Companion</h1>

<p align="center">
  <b>A FastAPI-based AI mental health chatbot</b><br>
  💬 Context-aware, empathetic responses | 🔥 Built with Ollama LLMs | 🧘 Crisis detection + conversation continuity
</p>



---

## ✨ Overview

**Sentiment-Analysis-driven Companion** is a production-ready mental health chatbot API built with **FastAPI** and **Ollama LLMs**. It goes beyond simple Q&A and provides:

- ✅ Emotionally intelligent support
- ✅ Conversation continuity using memory and context extraction
- ✅ Mental health crisis detection
- ✅ Streamed, contextual LLM responses
- ✅ Customizable system prompts and fine control via environment variables

---

## 🚀 Features

- 🧠 **Contextual Conversations** – Remembers what the user said and responds accordingly
- ❤️ **Empathetic Replies** – Crafted based on emotional progression and ongoing themes
- 🚨 **Crisis Detection** – Detects suicidal or harmful language and responds appropriately
- 🔐 **CORS + Rate Limiting** – API-safe for public deployment
- ⚙️ **Customizable Prompts and Models** – Easily configurable via `.env` or `os.environ`
- 🔌 **Ollama Integration** – Works with local LLMs like Mistral, LLaMA, or others via `ollama`

---

## 🛠️ Tech Stack

| Layer       | Technology        |
|-------------|-------------------|
| 🧠 Backend  | FastAPI + Python  |
| 🗣️ AI Engine | Ollama (Mistral by default) |
| 💬 NLP      | Custom prompt engineering, context extraction |
| 🔐 Security | CORS, rate limiting, validation |
| 📂 Logging  | Logging to console + file (`chatbot.log`) |

---

## 🧪 Example Workflow

1. Client sends a message via `/chat` endpoint.
2. Message is sanitized and analyzed for:
   - Sentiment
   - Crisis language
   - Ongoing topic context (e.g., job search, stress)
3. Context-aware prompt is generated.
4. Ollama model (default: Mistral) generates a thoughtful response.
5. Response is returned along with updated message history.

---

## 🧰 Usage

### 🔧 Installation

```bash
git clone https://github.com/Alfredd43/Sentiment-Analysis-driven-Companion.git
cd Sentiment-Analysis-driven-Companion

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt


# 🤖 PDF Chatbot with Gemini + Cohere + ChromaDB

This two-part Streamlit project allows you to upload a PDF, extract its **text**, **images**, and **tables**, summarize visual content using **Gemini**, embed using **Cohere**, and **chat with the document** using retrieval-augmented generation (RAG).

---

## 🧩 Components

### 1. `store.py` – Extract and Store
- Parses PDF into:
  - Clean text (via Unstructured)
  - Extracted tables (with HTML)
  - Extracted images (with surrounding context)
- Summarizes:
  - Tables using Gemini
  - Images using Gemini + context
- Embeds all content (text + summaries) using **Cohere v3**
- Saves to **ChromaDB**

### 2. `chat.py` – Chatbot UI
- Gemini-powered RAG chatbot
- Responds strictly using stored chunks (text/images/tables)
- Displays relevant sources below the answer:
  - Image previews
  - Table HTML
  - Source type and ID

---

## ⚙️ Setup

```bash
pip install -r requirements.txt

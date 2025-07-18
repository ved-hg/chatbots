# 📘 FAQ Collector & Gemini Chatbot with Memory

This is a Streamlit-based application that allows you to upload formatted FAQ-style PDFs, convert them into vector embeddings, and interact with them using Google's Gemini 2.5 Flash model via a chat interface. The chatbot maintains memory of past conversations and uses real-time document retrieval (RAG) to generate precise and grounded answers.

---

## 🔧 Features

- Upload FAQ PDFs and extract `Question: ... Answer: ...` pairs
- Embed text using `BAAI/bge-base-en-v1.5` via LangChain & Chroma
- Query embedded documents using Google's **Gemini 2.5 Flash**
- Maintains a searchable chat history with source attributions
- Real-time loading spinner and styled Q/A interface
- Clear chat and clear embedding functionalities

---

## 🖼️ UI Highlights

- Beautiful, styled chat bubbles for both user and Gemini bot
- Expanding source panels showing chunks used in the answer
- Spinner shows during Gemini response generation
- Upload panel for PDF with chunk-size slider and embedding button

---
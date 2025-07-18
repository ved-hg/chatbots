import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import cohere
import os
import json
import base64
import re
from PIL import Image

# Configuration
CHROMA_DIR = "chroma_storage"
IMAGE_DIR = "extracted_images"
COLLECTION_NAME = "pdf_data"
GEMINI_API_KEY = "your-gemini-key"

# Embedding Wrapper
class CohereEmbeddings:
    def __init__(self):
        self.client = cohere.Client("your-cohere-key")

    def embed_documents(self, texts):
        response = self.client.embed(
            texts=texts,
            model="embed-english-v3.0",
            input_type="search_document"
        )
        return response.embeddings

    def embed_query(self, text):
        return self.embed_documents([text])[0]

def generate_gemini_response(query, retriever, gemini_model, image_dir=IMAGE_DIR):
    docs = retriever.get_relevant_documents(query)
    contents = [
        {
            "text": f"""You are a helpful assistant. Answer the following user query using ONLY the provided content from documents, tables, and images.

User query: {query}

Instructions:
- Use only the provided content. Do NOT make up information.
- For text documents, rely on factual points.
- For images:
  - If an image supports your answer, mention:
    Image ID: <doc_id>
  - If the image is a graph, describe key trends and values.
- For tables:
  - Use both the summary and the HTML structure to derive your answer.
  - If specific values, columns, or relationships support your answer, refer to them clearly.
  - Mention Table ID: <doc_id> only if it directly supports the answer.
- Your answer must include at least one clear, factual statement.
- If unsure or unsupported, state that the documents do not contain a direct answer.

Output only your final answer. Do not repeat the input or metadata.
"""
        }
    ]

    # Include last 2–3 turns of chat history
    history_parts = []
    for past in st.session_state.chat_history[-3:]:
        history_parts.append({"text": f"Previous Q: {past['question']}\nA: {past['answer']}"})

    contents = history_parts + contents

    image_lookup = {}  # doc_id → image path

    for doc in docs:
        if doc.metadata["type"] == "text":
            contents.append({"text": doc.page_content})
        elif doc.metadata["type"] == "table":
            doc_id = doc.metadata["doc_id"]
            summary = doc.page_content.strip()
            table_html = doc.metadata.get("html", "").strip()

            contents.append({
                "text": f"### Table {doc_id}\nSummary: {summary}\n\nHTML Table:\n{table_html}"
            })
        elif doc.metadata["type"] == "image":
            doc_id = doc.metadata["doc_id"]
            img_path = os.path.join(image_dir, f"{doc_id}.png")
            image_lookup[doc_id] = img_path

            try:
                summary = json.loads(doc.metadata.get("summary", "{}"))
                contents.append({"text": f"### Image {doc_id}\nTitle: {summary.get('title', '')}\nInsights: {summary.get('insights', '')}"})
            except:
                contents.append({"text": f"Image {doc_id} summary could not be parsed."})

            if os.path.exists(img_path):
                with Image.open(img_path) as img:
                    w, h = img.size
                    contents.append({"text": f"Image ID: {doc_id} Image Size: ({w}, {h})"})
                with open(img_path, "rb") as f:
                    img_data = base64.b64encode(f.read()).decode("utf-8")
                contents.append({
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": img_data
                    }
                })

    # Generate Gemini response
    response = gemini_model.generate_content([{"parts": contents}])
    response_text = response.text.strip() if hasattr(response, "text") else "[No response]"

    st.markdown(f"**Answer:**\n\n{response_text}")

    # Detect which image Gemini referred to (if any)
    match = re.search(r"Image ID:\s*(img_\d+)", response_text)
    if match:
        doc_id = match.group(1)
        img_path = image_lookup.get(doc_id)
        if img_path and os.path.exists(img_path):
            st.markdown("### 🖼 Relevant Image Identified by Gemini:")
            st.image(img_path, caption=f"Full image from {doc_id}", use_column_width=False)
        else:
            st.warning(f"⚠️ Image for {doc_id} not found.")
    else:
        if image_lookup:
            fallback_img = next(iter(image_lookup.values()))
            st.markdown("### 🖼 Fallback Image Preview:")
            st.image(fallback_img, use_column_width=False)

    return response_text

# Initialize Gemini and Vector DB
def initialize_system():
    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=CohereEmbeddings(),
        collection_name=COLLECTION_NAME
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("models/gemini-2.5-flash")

    return retriever, gemini_model

# Display Sources
def display_source(doc):
    with st.expander(f"📄 Source ({doc.metadata['type'].upper()} {doc.metadata['doc_id']})"):
        if doc.metadata["type"] == "image":
            img_path = os.path.join(IMAGE_DIR, f"{doc.metadata['doc_id']}.png")
            if os.path.exists(img_path):
                st.image(img_path, use_column_width=True)
                try:
                    summary = json.loads(doc.metadata.get("summary", "{}"))
                    st.markdown("**🧠 Gemini Summary:**")
                    st.json(summary)
                except:
                    st.warning("⚠️ Could not load image summary.")
            else:
                st.error(f"⚠️ Image not found: {img_path}")
        else:
            st.text_area("📑 Text Content", value=doc.page_content, height=150)

# Streamlit UI
def main():
    st.set_page_config(page_title="📄 PDF RAG Chatbot", layout="wide")
    st.markdown("""
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stChatMessage { background-color: #f9f9f9; border-radius: 10px; padding: 0.75rem; }
        .stMarkdown h3 { margin-top: 1.5rem; color: #2d76d9; }
        </style>
    """, unsafe_allow_html=True)

    st.title("🤖 PDF Knowledge Assistant (Multimodal)")
    st.caption("Ask questions about documents — powered by Gemini + Cohere + ChromaDB")
    st.divider()

    if 'retriever' not in st.session_state or 'llm' not in st.session_state:
        with st.spinner("🔍 Initializing system..."):
            retriever, llm = initialize_system()
            st.session_state.retriever = retriever
            st.session_state.llm = llm
            st.success("✅ Knowledge base loaded!")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if st.button("🧹 Clear Chat History"):
        st.session_state.chat_history = []
        st.success("Chat history cleared.")

    query = st.chat_input("💬 Ask about the PDF content...")

    if query:
        with st.spinner("🧠 Generating response..."):
            try:
                response = generate_gemini_response(query, st.session_state.retriever, st.session_state.llm)

                with st.chat_message("user"):
                    st.markdown(f"**You:** {query}")

                with st.chat_message("assistant", avatar="📚"):
                    st.markdown(f"**Answer:**\n\n{response}")
                    st.divider()
                    st.subheader("📚 Reference Sources")
                    docs = st.session_state.retriever.get_relevant_documents(query)
                    for doc in docs:
                        display_source(doc)

                st.session_state.chat_history.append({
                    "question": query,
                    "answer": response
                })

            except Exception as e:
                st.error(f"❌ Query failed: {str(e)}")

if __name__ == "__main__":
    main()

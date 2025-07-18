import io, re, base64, fitz, pdfplumber
import streamlit as st
from PIL import Image as PILImage
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import CompositeElement, Image as UImage, Table as UTable, Table
from langchain.text_splitter import RecursiveCharacterTextSplitter
import cohere
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain.docstore.document import Document
import os
import shutil
import google.generativeai as genai
import json
import hashlib
from typing import List, Dict, Any

# Configure Gemini
API_KEY = "your-gemini-key"
genai.configure(api_key=API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# Initialize Cohere
cohere_client = cohere.Client("your-cohere-key")

# Prompts
IMAGE_PROMPT = (
    "You are a helpful assistant analyzing a technical or scientific image.\n"
    "Use the nearby text (provided as context) to infer the most relevant title for the image.\n"
    "If the image is a graph or chart, identify the X and Y axes clearly.\n"
    "Then provide up to 3 concise insights based on the image.\n\n"
    "Respond in strict JSON format as:\n"
    "{\n"
    '  "title": "generated from context",\n'
    '  "axes": {"x": "label or unit", "y": "label or unit"},\n'
    '  "insights": ["...", "...", "..."]\n'
    "}\n"
    "Do not repeat the context. Be accurate and brief."
)

TABLE_PROMPT = (
    "You are a helpful assistant. Summarize the following table in a brief manner.\n"
    "Start the summary with: 'This table has ...' and include important headings that is top of columns, take care of sub headings in tables as well.\n"
    "Do not repeat the title, and do not exceed 4 lines.\n"
    "Respond in JSON format:\n"
    "{\n"
    '  "summary": "This table has ...",\n'
    '  "headings": ["column1", "column2", ...]\n'
    "}\n"
)

st.set_page_config(page_title="📄 Advanced PDF Extractor", layout="wide")
st.title("📄 Intelligent PDF Processor with ChromaDB")

uploaded = st.file_uploader("📎 Upload PDF File", type=["pdf"])

CHROMA_DIR = "chroma_storage"
IMAGE_DIR = "extracted_images"
os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

class CohereEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = cohere_client.embed(
            texts=texts,
            model="embed-english-v3.0",
            input_type="search_document"
        )
        return response.embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

cohere_embeddings = CohereEmbeddings()

def chunk_text(texts: List[str], chunk_size: int = 2500, chunk_overlap: int = 500) -> List[str]:
    text_combined = "\n\n".join(texts)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n"]
    )
    return [doc.page_content for doc in splitter.create_documents([text_combined])]

def extract_pdf_elements(uploaded_file):
    file_data = uploaded_file.read()
    elements = partition_pdf(
        file=io.BytesIO(file_data),
        strategy="hi_res",
        extract_images_in_pdf=True,
        extract_image_block_types=["Image", "Table", "Figure", "Chart"],
        extract_image_block_to_payload=True,
        infer_table_structure=True,
       
        include_page_breaks=False,
    )

    texts, images, tables = [], [], []
    element_counts = {"text": 0, "image": 0, "table": 0}

    for idx, el in enumerate(elements):
        # Handle tables
        if isinstance(el, (UTable, Table)):
            try:
                table_html = el.metadata.text_as_html if hasattr(el.metadata, 'text_as_html') else ""
                table_text = str(el).strip()
                
                tables.append({
                    "text": table_text,
                    "html": table_html,
                    "element_type": "table"
                })
                element_counts["table"] += 1
                continue
            except Exception as e:
                st.warning(f"Table processing error: {str(e)}")
                continue

        # Handle images
        if isinstance(el, UImage):
            try:
                img_bytes = base64.b64decode(el.metadata.image_base64)
                img = PILImage.open(io.BytesIO(img_bytes))
                
                # Get surrounding text context
                context = []
                for offset in [-3, -2, -1, 1, 2, 3]:  # Wider context window
                    pos = idx + offset
                    if 0 <= pos < len(elements):
                        el2 = elements[pos]
                        if not isinstance(el2, (UImage, UTable, Table)):
                            text_candidate = str(el2).strip()
                            if text_candidate:
                                context.append(text_candidate)
                
                images.append({
                    "base64": el.metadata.image_base64,
                    "image_obj": img,
                    "hash": hashlib.md5(img_bytes).hexdigest(),
                    "context": context,
                    "element_type": "image"
                })
                element_counts["image"] += 1
                continue
            except Exception as e:
                st.warning(f"Image processing error: {str(e)}")
                continue

        # Handle composite elements (may contain nested images/text)
        if isinstance(el, CompositeElement):
            sub_els = getattr(el.metadata, 'orig_elements', [])
            has_text = any(not isinstance(s, (UImage, UTable, Table)) for s in sub_els)

            # Process any nested images
            for s in sub_els:
                if isinstance(s, UImage):
                    try:
                        img_bytes = base64.b64decode(s.metadata.image_base64)
                        img = PILImage.open(io.BytesIO(img_bytes))
                        
                        context = []
                        for offset in [ -2, -1, 1, 2]:
                            pos = idx + offset
                            if 0 <= pos < len(elements):
                                el2 = elements[pos]
                                if not isinstance(el2, (UImage, UTable, Table)):
                                    text_candidate = str(el2).strip()
                                    if text_candidate:
                                        context.append(text_candidate)
                        
                        images.append({
                            "base64": s.metadata.image_base64,
                            "image_obj": img,
                            "hash": hashlib.md5(img_bytes).hexdigest(),
                            "context": context,
                            "element_type": "image"
                        })
                        element_counts["image"] += 1
                    except Exception as e:
                        st.warning(f"Nested image processing error: {str(e)}")

            if has_text:
                texts.append(str(el).strip())
                element_counts["text"] += 1
        else:
            texts.append(str(el).strip())
            element_counts["text"] += 1

    # Initial extraction report
    st.subheader("📊 Initial Extraction Results")
    st.write(f"📝 Text elements: {element_counts['text']}")
    st.write(f"🖼️ Images extracted: {element_counts['image']}")
    st.write(f"📑 Tables extracted: {element_counts['table']}")
    st.markdown("---")

    return chunk_text(texts), images, tables

def summarize_tables(tables):
    if not tables:
        return []
    
    summaries = []
    for i, tbl in enumerate(tables):
        try:
            response = gemini_model.generate_content([
                {"text": TABLE_PROMPT},
                {"text": f"Table HTML:\n{tbl['html']}\n\nTable Text:\n{tbl['text']}"}
            ])
            
            # Clean up the response
            text = response.text.strip()
            if text.startswith("```json"):
                text = text[7:].strip("`\n ")
            elif text.startswith("```"):
                text = text[3:].strip("`\n ")
            
            summary = json.loads(text)
            summaries.append({
                "index": i+1,
                "summary": summary.get("summary", "No summary generated"),
                "headings": summary.get("headings", []),
                "html": tbl["html"],
                "text": tbl["text"]
            })
        except Exception as e:
            st.warning(f"Failed to summarize table {i+1}: {str(e)}")
            summaries.append({
                "index": i+1,
                "summary": f"Summary error: {str(e)}",
                "headings": [],
                "html": tbl["html"],
                "text": tbl["text"]
            })
    
    return summaries

def summarize_images(images):
    if not images:
        return []
    
    all_summaries = []
    batch_size = 10  # Reduced for reliability
    failed_images = []
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        try:
            contents = []
            for img in batch:
                context_str = "\n".join(img.get("context", []))[:1000]
                parts = [
                    {"text": f"{IMAGE_PROMPT}\n\nContext:\n{context_str}"},
                    {"inline_data": {
                        "mime_type": "image/png",
                        "data": img["base64"]
                    }}
                ]
                contents.append({"role": "user", "parts": parts})
            
            # Get batch response
            response = gemini_model.generate_content(contents)
            
            # Process each response individually with error handling
            for j, (img, candidate) in enumerate(zip(batch, response.candidates)):
                img_index = i + j + 1
                try:
                    if not candidate.content.parts:
                        raise ValueError("Empty response from Gemini")
                        
                    text = candidate.content.parts[0].text.strip()
                    
                    # Clean JSON response
                    text = text.replace('```json', '').replace('```', '').strip()
                    
                    # Handle multiple JSON objects case
                    if text.count('{') > 1:
                        text = text[text.find('{'):text.rfind('}')+1]
                    
                    summary = json.loads(text)
                    
                    all_summaries.append({
                        "index": img_index,
                        "summary": summary,
                        "base64": img["base64"],
                        "hash": img["hash"],
                        "image_obj": img["image_obj"],
                        "context": img["context"]
                    })
                    
                except Exception as e:
                    failed_images.append(img_index)
                    st.warning(f"Failed to parse image {img_index} summary: {str(e)}")
                    all_summaries.append({
                        "index": img_index,
                        "summary": {
                            "title": f"Image {img_index} (Parse Error)",
                            "axes": {"x": "Unknown", "y": "Unknown"},
                            "insights": [f"Original error: {str(e)[:100]}"]
                        },
                        "base64": img["base64"],
                        "hash": img["hash"],
                        "image_obj": img["image_obj"],
                        "context": img["context"]
                    })
                    
        except Exception as batch_error:
            st.error(f"Batch {i//batch_size + 1} failed: {str(batch_error)}")
            for j, img in enumerate(batch):
                img_index = i + j + 1
                failed_images.append(img_index)
                all_summaries.append({
                    "index": img_index,
                    "summary": {
                        "title": f"Image {img_index} (Batch Error)",
                        "axes": {"x": "Unknown", "y": "Unknown"},
                        "insights": [f"Batch processing failed: {str(batch_error)[:100]}"]
                    },
                    "base64": img["base64"],
                    "hash": img["hash"],
                    "image_obj": img["image_obj"],
                    "context": img["context"]
                })
    
    if failed_images:
        st.warning(f"Summary generation failed for {len(failed_images)} images: {', '.join(map(str, failed_images))}")
    
    return all_summaries
def store_embeddings(text_chunks, image_summaries, table_summaries):
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=cohere_embeddings,
        collection_name="pdf_data"
    )

    docs = []
    stored_counts = {"text": 0, "image": 0, "table": 0}

    # Store text chunks
    for i, chunk in enumerate(text_chunks):
        docs.append(Document(
            page_content=chunk,
            metadata={"type": "text", "doc_id": f"text_{i}"}
        ))
        stored_counts["text"] += 1

    # Store image summaries
    for i, img in enumerate(image_summaries):
        try:
            summary = img.get("summary", {})
            title = summary.get("title", "") or "Untitled Image"
            insights = summary.get("insights", []) or ["No insights provided."]
            
            content = f"{title}\nKey Insights:\n" + "\n".join(f"- {insight}" for insight in insights)
            
            docs.append(Document(
                page_content=content,
                metadata={
                    "type": "image",
                    "doc_id": f"img_{i}",
                    "hash": img["hash"],
                    "context": "\n".join(img.get("context", []))[:500]
                }
            ))
            stored_counts["image"] += 1
        except Exception as e:
            st.warning(f"Failed to embed image {i}: {str(e)}")

    # Store table summaries
    for i, tbl in enumerate(table_summaries):
        try:
            content = f"{tbl.get('summary', 'No summary')}\nHeadings: {', '.join(tbl.get('headings', []))}"
            
            docs.append(Document(
                page_content=content,
                metadata={
                    "type": "table",
                    "doc_id": f"tbl_{i}",
                    "html": tbl.get("html", ""),
                    "text": tbl.get("text", "")[:500]
                }
            ))
            stored_counts["table"] += 1
        except Exception as e:
            st.warning(f"Failed to embed table {i}: {str(e)}")

    vectordb.add_documents(docs)
    vectordb.persist()
    
    # Final storage report
    st.subheader("📊 Final Storage Summary")
    st.write(f"✅ Text chunks stored: {stored_counts['text']}")
    st.write(f"🖼️ Image summaries stored: {stored_counts['image']}")
    st.write(f"📑 Table summaries stored: {stored_counts['table']}")
    st.markdown("---")

if uploaded:
    with st.spinner("⏳ Processing your PDF..."):
        try:
            # Extract content
            chunks, images, tables = extract_pdf_elements(uploaded)
            
            # Summarize content
            with st.spinner("🔍 Summarizing images..."):
                image_summaries = summarize_images(images)
                
            with st.spinner("📊 Summarizing tables..."):
                table_summaries = summarize_tables(tables)
            
            # Store in ChromaDB
            with st.spinner("💾 Storing embeddings..."):
                store_embeddings(chunks, image_summaries, table_summaries)

            # Display extracted content
            st.subheader("📑 Extracted Tables")
            for i, tbl in enumerate(table_summaries):
                with st.expander(f"Table {tbl['index']}: {tbl['summary'][:50]}..."):
                    st.markdown(tbl["html"], unsafe_allow_html=True)
                    st.json({
                        "summary": tbl["summary"],
                        "headings": tbl["headings"]
                    })

            st.subheader("🖼️ Extracted Images")
            for img_summary in image_summaries:
                summary = img_summary["summary"]
                with st.expander(f"Image {img_summary['index']}: {summary.get('title', 'Untitled')}"):
                    st.image(base64.b64decode(img_summary["base64"]), 
                           caption=summary.get('title', ''), 
                           use_column_width=True)
                    st.markdown(f"**Axes:** X = `{summary.get('axes', {}).get('x', 'N/A')}`, Y = `{summary.get('axes', {}).get('y', 'N/A')}`")
                    st.markdown("**Insights:**")
                    for ins in summary.get("insights", []):
                        st.markdown(f"- {ins}")
                    if img_summary.get("context"):
                        st.markdown("**Nearby Text Context:**")
                        st.text("\n".join(img_summary["context"]))

        except Exception as e:
            st.error(f"❌ PDF processing failed: {str(e)}")
            st.exception(e)

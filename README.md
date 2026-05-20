# RAG with Vector Stores & Embeddings

A hands-on implementation of **Retrieval-Augmented Generation (RAG)** using Google Gemini embeddings, LangChain, and two vector store backends - **ChromaDB** and **FAISS**. The pipeline ingests a PDF document, chunks and embeds it, stores the vectors, and retrieves semantically relevant content in response to natural language queries.

---

## What This Project Does

Traditional LLMs answer questions from memory - limited by their training data. This project builds a RAG pipeline that:

1. **Loads** a PDF document (`attention.pdf` - the seminal *"Attention Is All You Need"* paper)
2. **Splits** it into overlapping chunks for better context preservation
3. **Embeds** each chunk using Google's `gemini-embedding-001` model
4. **Stores** the embeddings in a vector database (Chroma or FAISS)
5. **Retrieves** the most semantically relevant chunks for any user query via similarity search

---

## Tech Stack

| Component | Tool / Library |
|---|---|
| Embeddings | `google-generativeai` · `gemini-embedding-001` |
| LLM integration | `langchain-google-genai` · `ChatGoogleGenerativeAI` |
| Document loading | `langchain-community` · `PyPDFLoader` |
| Text splitting | `langchain` · `RecursiveCharacterTextSplitter` |
| Vector store (option 1) | `ChromaDB` |
| Vector store (option 2) | `FAISS` (CPU) |

---

## Project Structure

```
RAGwithVectorandEmbeddings/
│
├── RAGwithVectorandEmbeddings.ipynb   # Main notebook
├── attention.pdf                       # Source document (Vaswani et al., 2017)
└── README.md
```

---

## Pipeline Walkthrough

### Step 1 - Load the PDF
```python
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader('attention.pdf')
docs = loader.load()
```

### Step 2 - Chunk the text
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(docs)
```
`chunk_overlap=200` ensures context isn't lost at chunk boundaries.

### Step 3 - Embed with Gemini
```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=google_api_key
)
```

### Step 4 - Store in ChromaDB
```python
from langchain_community.vectorstores import Chroma
db = Chroma.from_documents(documents, embedding_model)
```

### Step 5 - Store in FAISS (alternative)
```python
from langchain_community.vectorstores import FAISS
db1 = FAISS.from_documents(documents[:15], embedding_model)
```

### Step 6 - Query with similarity search
```python
query = "Who are the authors of attention is all you need?"
results = db.similarity_search(query)
print(results[0].page_content)
```

---

## Chroma vs FAISS

| Feature | ChromaDB | FAISS |
|---|---|---|
| Persistence | Built-in (disk-backed) | Manual save/load |
| Ease of use | Simple, great for prototyping | Faster at scale |
| Filtering | Metadata filtering supported | Limited |
| Best for | Development & small datasets | High-performance retrieval |

---

## Example Query

```
Query: "Who are the authors of attention is all you need?"

Retrieved: "Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin..."
```

---

## Potential Extensions

- Add a conversational QA chain using `ChatGoogleGenerativeAI` for full end-to-end RAG
- Persist the Chroma vector store to disk for reuse across sessions
- Support multiple PDFs or web-scraped documents
- Build a Gradio or Streamlit front-end for interactive querying
- Swap the embedding model (e.g. OpenAI, HuggingFace) for comparison

---

## Requirements

- Python 3.9+
- A valid [Google AI Studio API key](https://aistudio.google.com/app/apikey)
- Jupyter Notebook or JupyterLab

---

This is a hands-on exploration of LLM application development with a focus on practical, real-world use cases.
The `attention.pdf` source document is the property of its respective authors (Vaswani et al., 2017) and is used here solely for demonstration.

---

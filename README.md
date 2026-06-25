# Multi-PDF RAG Chatbot 📚🤖

A conversational Retrieval-Augmented Generation (RAG) application built using Streamlit, LangChain, and Groq. This application allows users to upload multiple PDF documents simultaneously, process them into vector embeddings, and have context-aware, multi-turn conversations about the contents of those documents.

---

## 🚀 Features
* **Multi-PDF Support:** Upload and process multiple PDF documents at once.
* **Intelligent Text Chunking:** Dynamically breaks down long documents to fit LLM context limits while preserving semantics.
* **Local Vector Storage:** Uses `sentence-transformers` and **FAISS** for fast, local similarity searches.
* **True Conversational Memory:** Leverages `ConversationBufferMemory` to maintain full chat history, allowing you to ask natural follow-up questions.
* **Blazing Fast LLM Responses:** Powered by Meta's `llama-3.1-8b-instant` via the **Groq API**.

---

## 🛠️ Architecture Workflow

1. **Ingestion:** Documents are parsed using `PyPDF2` and split into smaller chunks (1,000 characters with a 300-character overlap).
2. **Embedding & Indexing:** Chunks are transformed into vectors using the `all-MiniLM-L6-v2` model and saved locally in a FAISS database.
3. **Question Contextualization:** When you submit a follow-up query, the chain utilizes past chat history to rewrite your input into a standalone, clear question.
4. **Retrieval & Generation:** The standalone question retrieves relevant document blocks from FAISS, and the Groq LLM generates the final answers.

---

## 📦 Installation & Setup

Follow these steps to get the application running locally on your machine:

### 1. Clone the Repository
```bash
git clone [https://github.com/Autumsjoy/multi-pdf-rag-chatbot]
cd multi-pdf-rag-chatbot

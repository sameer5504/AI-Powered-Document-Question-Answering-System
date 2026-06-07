# 📚 DocuMind - AI PDF Study Assistant

DocuMind is an AI-powered PDF Study Assistant that allows users to upload documents and interact with them using natural language. The system uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from PDFs and generate context-aware answers.

## Features

* 📄 Upload and analyze multiple PDF documents
* 🔍 Semantic search using vector embeddings
* 🤖 AI-generated answers using local Llama 3 models through Ollama
* 📚 Source citation with page references
* 💬 Conversation history
* ⚡ FAISS vector database for fast retrieval
* 🧠 Sentence Transformer embeddings
* 📊 Retrieval evaluation dashboard
* ⚙️ Adjustable chunk size, overlap, and retrieval settings

## Tech Stack

* Python
* Streamlit
* LangChain
* FAISS
* Sentence Transformers
* Ollama
* Llama 3

## Architecture

PDF Upload → Text Extraction → Chunking → Embeddings → FAISS Vector Store → Semantic Search → Llama 3 → Answer Generation

## Installation

### Clone the repository

```bash
git clone https://github.com/yourusername/DocuMind.git
cd DocuMind
```

### Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment:

Windows:

```bash
.venv\Scripts\activate
```

Linux/Mac:

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Install Ollama (Optional)

Download Ollama:

https://ollama.com

Pull a model:

```bash
ollama pull llama3.2:1b
```

or

```bash
ollama pull llama3
```

## Running the Application

```bash
streamlit run app.py
```

## Usage

1. Upload one or more PDF documents.
2. Build the vector database.
3. Ask questions about the uploaded documents.
4. Review generated answers and source citations.

Example questions:

* What is referential integrity?
* Explain Kantianism in simple words.
* Summarize the uploaded PDFs.
* Generate exam questions from these documents.

## Project Highlights

This project demonstrates:

* Retrieval-Augmented Generation (RAG)
* Natural Language Processing (NLP)
* Vector Databases
* Semantic Search
* Large Language Models (LLMs)
* Document Question Answering Systems

## Future Improvements

* PDF summarization
* Quiz generation
* Multi-document comparison
* Web deployment
* Support for additional file formats
* Hybrid keyword + vector search

## Author
## Author

**Samir Ali**

Computer Engineering Student

Developed an intelligent document question-answering system that leverages Retrieval-Augmented Generation (RAG), semantic search, vector embeddings, and local Large Language Models (LLMs) to provide accurate, context-aware responses from unstructured PDF documents. The system integrates LangChain, FAISS, Sentence Transformers, and Ollama to enable efficient document retrieval, conversational querying, and source-grounded answer generation.

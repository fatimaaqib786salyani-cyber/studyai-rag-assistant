# StudyAI — RAG Study Assistant

StudyAI is an AI-powered study assistant that transforms PDF notes into an interactive learning workspace.

Upload a PDF and use four study modes:

- **Ask Notes** — ask questions and receive answers grounded in the uploaded document
- **Smart Summary** — generate a structured revision guide
- **Interactive Quiz** — create multiple-choice questions, submit answers, and view scores with explanations
- **Revision Flashcards** — generate concise question-and-answer cards from important concepts

## How It Works

StudyAI uses a Retrieval-Augmented Generation (RAG) pipeline:

```text
PDF Upload
    ↓
Text Extraction
    ↓
Text Chunking
    ↓
Sentence Transformer Embeddings
    ↓
FAISS Vector Index
    ↓
User Question
    ↓
Similarity Search
    ↓
Relevant Document Context
    ↓
Groq LLM
    ↓
Grounded Answer
```

The Q&A workflow retrieves relevant passages from the uploaded PDF before generating an answer. If the requested information is not supported by the retrieved notes, the assistant is instructed not to invent an answer.

## Tech Stack

- Python 3.11
- Streamlit
- PyPDF
- Sentence Transformers
- `all-MiniLM-L6-v2` embeddings
- FAISS
- Groq API
- `openai/gpt-oss-120b`

## Project Structure

```text
ai-study-assistant/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    └── secrets.toml
```

> `.streamlit/secrets.toml` contains the API key and must not be committed to GitHub.

## Run Locally

### 1. Create a virtual environment

```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Configure the Groq API key

Create:

```text
.streamlit/secrets.toml
```

Add:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

### 4. Start StudyAI

```bash
python -m streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Study Modes

### Ask Notes
Uses semantic retrieval with FAISS to find relevant chunks before generating a document-grounded response. Supporting passages and source pages can be inspected in the interface.

### Smart Summary
Turns the uploaded document into a structured study guide with an overview, key concepts, important details, and quick revision points.

### Interactive Quiz
Generates five multiple-choice questions from the notes. Students answer the questions first and then receive a score and explanations.

### Revision Flashcards
Generates a revision deck containing concise questions and answers based on important concepts from the uploaded material.

## Privacy and API Keys

Never commit API credentials to the repository.

Recommended `.gitignore` entries:

```gitignore
venv/
.streamlit/secrets.toml
__pycache__/
*.pyc
.DS_Store
.env
```

For Streamlit Community Cloud, add `GROQ_API_KEY` through the app's Secrets settings instead of uploading `secrets.toml`.

## Current Limitations

- The current interface accepts one PDF at a time.
- Text extraction works best with text-based PDFs; scanned/image-only PDFs may require OCR.
- The FAISS index is created in memory for the active document and is not a persistent database.
- Generated output quality depends on the content and structure of the uploaded notes.

## Purpose

This project demonstrates how Retrieval-Augmented Generation can be used to build an educational assistant that combines semantic search, vector retrieval, and large language models while keeping responses grounded in user-provided study material.

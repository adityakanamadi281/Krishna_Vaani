# Krishnavani_GPT
## उद्धरेदात्मनात्मानं नात्मानमवसादयेत् ।
## आत्मैव ह्यात्मनो बन्धुरात्मैव रिपुरात्मनः ॥

## Uddharedātmanātmānaṁ nātmānam avasādayet | Ātmaiva hyātmano bandhur ātmaiva ripur ātmanaḥ || 
![krishna](https://github.com/user-attachments/assets/e4314b8d-3e1c-40b7-b742-82b9a19b13e0)

---

## Overview

**Krishna Vaani (Krishnavani_GPT)** is a production-grade Retrieval-Augmented Generation (RAG) system built to answer questions, doubts, and seekers of guidance with the timeless wisdom of the **Bhagavad Gita**. The system is designed to provide highly grounded, cited, and context-accurate answers in the serene and compassionate persona of Lord Krishna.

It uses a robust document parsing pipeline, semantic chunking, a hybrid retrieval system (dense FAISS vector retrieval + sparse BM25 keyword retrieval), cross-encoder reranking (via FlashRank), and localized generative LLM execution (via Ollama).

---

## Tech Stack

- **Core / Programming Language**: Python (>= 3.10)
- **RAG & LLM Framework**: LangChain 
- **Vector Database**: FAISS (CPU-based indexing)
- **Embeddings**: FastEmbed (using `sentence-transformers/all-MiniLM-L6-v2`)
- **Reranker**: FlashRank (using `ms-marco-MiniLM-L-12-v2`)
- **LLM Engine**: Ollama (configured with `qwen3.5:0.8b` or others)
- **Document Parsing**: PyMuPDF (`fitz`) for structure-aware parsing
- **CLI & Configurations**: Typer for command-line interface, Pydantic Settings for configuration management, and Loguru for structured logging.

---

## Project Structure

```text
Krishna_Vaani/
├── data/
│   └── The Bhagavad Gita.pdf      # The source text of Bhagavad Gita (PDF)
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py            # Pydantic settings configuration loader
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── generator.py           # Response generation and streaming using Ollama
│   │   └── prompts.py             # System prompts for query rewriting and persona response
│   ├── indexing/
│   │   ├── __init__.py
│   │   └── vector_store.py        # FAISS index building and loading manager
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── chunker.py             # Semantic chunking that preserves verse context
│   │   └── parser.py              # Custom PDF parser targeting chapters, verses, and comments
│   └── retrieval/
│       ├── __init__.py
│       ├── hybrid.py              # Hybrid dense-sparse search combining FAISS & BM25
│       └── reranker.py            # FlashRank CPU cross-encoder reranking layer
├── .env.example                   # Configuration template for paths, models, and servers
├── pyproject.toml                 # Package description and dependency requirements
├── run.py                         # Main CLI script containing `ingest` and `query` commands
└── README.md                      # Project documentation (this file)
```

---

## Setup & Installation

### 1. Prerequisites
- **Python**: Ensure you have Python 3.10 or higher installed.
- **Ollama**: Download and install Ollama from [ollama.ai](https://ollama.ai/).

### 2. Install Dependencies
You can install dependencies into your environment using `uv` (recommended):

```bash
# Sync dependencies and set up the virtual environment
uv sync
```

### 3. Environment Setup
Copy the example configuration to a `.env` file and verify your paths and models:

```bash
copy .env.example .env
```

### 4. Setup the LLM
Start Ollama on your machine and pull the default Qwen model:

```bash
ollama pull qwen3.5:0.8b
```

### 5. Ingest the Book
Parse and build the vector database indices from the PDF document:

```bash
uv run python run.py ingest
```

### 6. Interactive Conversation Chat
To start a continuous, back-and-forth conversation session with conversational memory and real-time response streaming, run the following command:

```bash
uv run python run.py chat
```

Use the `--verbose` or `-v` flag if you wish to see background ingestion, retrieval, and reranking logs during your chat session:
```bash
uv run python run.py chat -v
```

> [!NOTE]
> Alternatively, you can activate the virtual environment and run python commands directly:
> ```bash
> # Windows (PowerShell)
> .venv\Scripts\Activate.ps1
> python run.py chat
> 
> # macOS / Linux
> source .venv/bin/activate
> python run.py chat
> ```

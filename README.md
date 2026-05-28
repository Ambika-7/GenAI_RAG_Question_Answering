# GenAI RAG Question Answering

This repository contains a modular Retrieval-Augmented Generation (RAG) Question Answering pipeline built using LangChain, HuggingFace embeddings, and local LLMs via Ollama.

## Features
- **Modular RAG pipeline**: Separate modules for document loading, chunking, embeddings, and vector stores.
- **Multiple Vector Stores**: Support for both FAISS and Chroma.
- **Local LLMs**: Powered by Ollama, supporting models like Llama 3, Gemma, etc.
- **Interactive QA**: Run `app.py` to chat with your documents.
- **Evaluation Framework**: Run `evaluate.py` to measure performance using ROUGE, BLEU, BERTScore, and accuracy.

## Prerequisites
- Python 3.8+
- [Ollama](https://ollama.com/) installed and running locally.

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Ambika-7/GenAI_RAG_Question_Answering.git
   cd GenAI_RAG_Question_Answering
   ```

2. **Create a virtual environment** (Optional but recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   # source venv/bin/activate
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Pull the necessary LLM models** via Ollama:
   ```bash
   ollama pull llama3
   ollama pull gemma
   ```

## Usage

### 1. Interactive App
Make sure you have a sample PDF document located at `data/documents/healthcare.pdf` (or modify `PDF_PATH` in `app.py`).

Run the following command to start an interactive Question-Answering loop:
```bash
python app.py
```

### 2. Evaluation
To run evaluations on different experiments, ensure you have your test dataset at `data/qa_pairs/test_data.csv`. Then run:
```bash
python evaluate.py
```
This will output metrics such as ROUGE, BLEU, BERTScore, and Accuracy based on your Ground Truth data.

# AI Study Companion

An AI-powered tool that summarizes lecture or meeting transcripts into a clean summary with action items, and lets you ask follow-up questions grounded in the original content using Retrieval-Augmented Generation (RAG).

## Why I built this

As a final-year student, I often have long lecture transcripts and notes that are tedious to revise from. This tool automatically extracts the key points and action items, and lets me quickly find specific information by just asking a question — instead of manually re-reading everything.

## How it works

1. **Summarization** — the full transcript is sent to Google's Gemini model with a structured prompt to generate a concise summary and a list of action items.
2. **Retrieval-Augmented Generation (RAG)** — the transcript is split into overlapping chunks, converted into vector embeddings using a local sentence-transformers model, and stored in a ChromaDB vector database. When a question is asked, the most relevant chunks are retrieved and passed to Gemini along with the question, so answers are grounded in the actual transcript instead of the model guessing.

## Tech stack

- Python
- Google Gemini API (`google-genai`) for summarization and question answering
- `sentence-transformers` for local, free text embeddings
- ChromaDB as the vector database for semantic search
- Streamlit for the web interface

## Key design decisions

- **Word-boundary chunking**: chunks are built by adding whole words until a size limit is reached, rather than cutting text at fixed character counts, to avoid splitting words or sentences awkwardly.
- **Overlap between chunks**: consecutive chunks share some content, so information near chunk boundaries isn't lost during retrieval.
- **Hallucination prevention**: the Q&A prompt explicitly instructs the model to say "I don't have enough information" if the answer isn't in the retrieved context, rather than guessing.

## Running it locally

1. Clone this repo and install dependencies:
```bash
   pip install -r requirements.txt
```
2. Create a `.env` file with your Gemini API key:
GEMINI_API_KEY=your_key_here
3. Run the app:
```bash
   streamlit run app.py
```

## What I'd improve with more time

- Support for uploading audio and auto-transcribing it (currently text-only)
- Better handling of very long transcripts (splitting into multiple summarization passes)
- Persisting the vector database so previously processed transcripts don't need to be re-indexed
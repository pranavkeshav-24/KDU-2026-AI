# Content Accessibility Suite

Streamlit MVP for converting PDFs, images, and audio into accessible text, summaries, searchable chunks, hybrid search results, and transparent cost logs.

## Features

- PDF upload with local PyMuPDF page inventory before vision enrichment.
- Selective vision policy for scanned or mixed visual pages.
- Image OCR and accessibility descriptions through OpenAI vision when configured.
- Local Whisper transcription path for audio files.
- Unified document representation with summaries, key points, topic tags, and accessibility notes.
- SQLite metadata, outputs, chunk storage, FTS5 keyword search, local vector persistence, and reciprocal-rank hybrid search.
- Cost dashboard for LLM, vision, embedding, local operations, and query logs.
- Local fallbacks when API keys or optional AI packages are unavailable.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Add `OPENAI_API_KEY` to `.env` to enable OpenAI summaries, embeddings, and vision. If OpenAI is not configured, `OPENROUTER_API_KEY` can be used for chat and vision-compatible fallback models. Without either key, the app still runs with local summaries and hash-based vectors.

## Run

```powershell
streamlit run app.py
```

## Test

```powershell
pytest
```

## Project Structure

- `app.py`: Streamlit UI.
- `src/orchestrator.py`: End-to-end processing workflow.
- `src/processors/`: PDF, image, and audio processors.
- `src/layout/`: PDF inventory, classification, and visual region detection.
- `src/retrieval/`: Chunking, embeddings, vector search, keyword search, and hybrid fusion.
- `src/storage/`: SQLite schema and data access.
- `src/cost/`: Pricing and cost calculation.
- `data/`: Local uploads, processed outputs, renders, vectors, and database.

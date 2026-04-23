# Multi-Function AI Assistant (OpenAI Only)

Production-style chatbot with:

1. Tool calling (Weather, Calculator, Search)
2. Streaming responses
3. Token usage and cost tracking
4. Streamlit web interface

## Files

- `app.py`: Streamlit app with OpenAI chat + tool orchestration
- `tools.py`: Tool implementations and function schemas
- `.env.example`: Environment template

## Environment

Create `.env` in the project root:

```env
OPENAI_API_KEY=your_real_openai_key
SERPER_API_KEY=your_serper_key
OPENAI_MODEL=gpt-5-nano-2025-08-07
OPENAI_INPUT_COST_PER_1M=0.05
OPENAI_OUTPUT_COST_PER_1M=0.40
```

Notes:

- OpenRouter is intentionally not used.
- The app loads `.env` from the repo root (same folder as `app.py`).

## Run

PowerShell:

```powershell
cd C:\Users\Dell\Documents\GitHub\KDU-2026-AI
python -m streamlit run app.py
```

If you use a venv:

```powershell
cd C:\Users\Dell\Documents\GitHub\KDU-2026-AI
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Troubleshooting 401

1. Make sure you are running `app.py` (not any older Streamlit file).
2. Ensure `OPENAI_API_KEY` is set to a real key (not placeholder text).
3. Restart Streamlit after editing `.env`.
4. In the app sidebar, paste key manually once to verify auth works.

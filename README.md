# Multimodal AI Assistant

A robust, context-aware, multimodal AI assistant leveraging the LangChain framework. It queries models across the OpenRouter API gateway and is currently configured for the live Gemma 4 free-tier models.

## Setup & Run Locally

1. **Install Python & Deps**
   Ensure Python 3.11+ is installed.
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Keys & Configuration**
   Copy `.env.example` to `.env`. Ensure you map the OpenRouter and OpenWeatherMap keys!
   ```env
   OPENROUTER_API_KEY=sk-or-v1-xxx
   OPENWEATHERMAP_API_KEY=xxx
   REDIS_URL=redis://localhost:6379
   DATABASE_URL=sqlite:///./multimodal_assistant.db
   OPENROUTER_SITE_URL=http://127.0.0.1:8001
   OPENROUTER_APP_NAME=KDU-2026-AI
   ```

3. **Start Redis Context Memory (Docker needed)**
   ```bash
   docker-compose up -d redis
   ```

4. **Launch Application**
   ```bash
   uvicorn main:app --reload
   ```

5. **Open the Frontend**
   Visit `http://127.0.0.1:8000/` in the browser. Opening `/chat` in the browser will redirect back to the UI.

If Redis or external API keys are missing, the UI still loads and the backend returns a structured fallback response instead of crashing.

## Features

* **Multimodal Image Sight**: Pass `bytes` or multipart binaries pointing to the `/chat/image` endpoint to analyze photos contextually using `google/gemma-4-26b-a4b-it:free`.
* **Dynamic Modalities**: Switch persona explicitly via passing the HTTP Header `X-Response-Style` (`expert`, `casual`, `child`, `formal`).
* **Tool Bindings**: Asks LangChain strictly for autonomous data like native integration into `openweathermap` bypassing the standard hallucination responses.
* **Persistent Cache**: Scoped `thread_id` sessions are saved natively in Redis matching up to 24-hours to save on token context load bounds across restarts.

## Running Tests

Pytest provides an isolated view mocking Redis and guaranteeing unit validation bounds.
```bash
pytest -q tests
```

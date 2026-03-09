# FastAPI LLM API

A FastAPI application with LLM-powered text summarization and sentiment analysis, deployed on Render.

## Endpoints

- `GET /health` — Health check with timestamp
- `POST /summarize` — Summarize text with configurable max length and 3 prompt variations
- `POST /analyze-sentiment` — Analyze text sentiment with confidence score and 3 prompt variations

## Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file:
```
OPENAI_API_KEY=your-key-here
```

## Run
```bash
uvicorn app.main:app --reload
```

API docs at `http://127.0.0.1:8000/docs`

## Prompt Variations

Each LLM endpoint supports 3 prompt engineering techniques via the `prompt_version` parameter:
- Version 1: Basic direct instruction
- Version 2: Few-shot (with examples)
- Version 3: Chain-of-thought (step-by-step reasoning)



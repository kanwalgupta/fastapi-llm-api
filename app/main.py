import os
import json
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="LLM API",
    description="FastAPI application with LLM-powered summarization and sentiment analysis.",
    version="1.0.0",
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class SummarizeRequest(BaseModel):
    text: str
    max_length: int = 100


class SummarizeResponse(BaseModel):
    summary: str


class SentimentRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    explanation: str


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(request: SummarizeRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            messages=[
                {
                    "role": "system",
                    "content": f"You are a precise summarizer. Summarize the given text in no more than {request.max_length} words. Be concise and capture the key points. Return only the summary, nothing else."
                },
                {
                    "role": "user",
                    "content": request.text
                }
            ]
        )
        summary = response.choices[0].message.content.strip()
        return SummarizeResponse(summary=summary)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")


@app.post("/analyze-sentiment", response_model=SentimentResponse)
def analyze_sentiment(request: SentimentRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": 'You are a sentiment analysis expert. Analyze the sentiment of the given text. Respond in exactly this JSON format and nothing else: {"sentiment": "positive" or "negative" or "neutral", "confidence": 0.0 to 1.0, "explanation": "brief explanation"}'
                },
                {
                    "role": "user",
                    "content": request.text
                }
            ]
        )
        result = response.choices[0].message.content.strip()
        parsed = json.loads(result)

        return SentimentResponse(
            sentiment=parsed["sentiment"],
            confidence=parsed["confidence"],
            explanation=parsed["explanation"],
        )

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="LLM returned invalid format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

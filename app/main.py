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
    prompt_version: int = 1



class SummarizeResponse(BaseModel):
    summary: str


class SentimentRequest(BaseModel):
    text: str
    prompt_version: int = 1

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

    prompts = {
        1: f"You are a precise summarizer. Summarize the given text in no more than {request.max_length} words. Be concise and capture the key points. Return only the summary, nothing else.",
        2: f"""You are a precise summarizer. Here is an example:

Text: "The Amazon rainforest produces approximately 20% of the world's oxygen. It spans across nine countries in South America and is home to over 10 million species. Deforestation has reduced its size by 17% in the last 50 years, threatening global climate stability."
Summary: "The Amazon rainforest, spanning nine countries with over 10 million species, produces 20% of global oxygen but has lost 17% of its area to deforestation, threatening climate stability."

Now summarize the following text in no more than {request.max_length} words. Return only the summary.""",
        3: f"""You are a precise summarizer. Follow these steps:
1. Identify the main topic of the text
2. Extract the 3 most important facts or arguments
3. Combine them into a single coherent summary of no more than {request.max_length} words

Return only the final summary, not the steps."""
    }

    system_prompt = prompts.get(request.prompt_version, prompts[1])

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.text}
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

    prompts = {
        1: 'You are a sentiment analysis expert. Analyze the sentiment of the given text. Respond in exactly this JSON format and nothing else: {"sentiment": "positive" or "negative" or "neutral", "confidence": 0.0 to 1.0, "explanation": "brief explanation"}',
        2: """You are a sentiment analysis expert. Here are examples:

Text: "This product is amazing, best purchase I ever made!"
{"sentiment": "positive", "confidence": 0.95, "explanation": "Strong positive language with superlatives indicating high satisfaction"}

Text: "The delivery was late and the item was broken."
{"sentiment": "negative", "confidence": 0.9, "explanation": "Reports two negative experiences with no positive aspects"}

Text: "The package arrived on Tuesday."
{"sentiment": "neutral", "confidence": 0.85, "explanation": "Purely factual statement with no emotional indicators"}

Now analyze the following text. Respond in the same JSON format only.""",
        3: """You are a sentiment analysis expert. Analyze the given text by following these steps internally:
1. Identify emotionally charged words
2. Determine if the overall tone is positive, negative, or neutral
3. Assess how confident you are based on the strength of the language

Respond in exactly this JSON format and nothing else: {"sentiment": "positive" or "negative" or "neutral", "confidence": 0.0 to 1.0, "explanation": "brief explanation"}"""
    }

    system_prompt = prompts.get(request.prompt_version, prompts[1])

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
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

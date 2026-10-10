"""OpenAI Decisions client: sends one message with all questions and returns a normalized result.

Run:  uv run openai_client.py  ->  asks the first 3 sample rows
"""

import asyncio
import time

from dotenv import load_dotenv
from openai import AsyncOpenAI

from dataset import QUESTIONS, load_sample

load_dotenv()

MODEL = "gpt-6-luna"
PRICE_PER_M_INPUT = 0.10  # USD per 1M input tokens, output is free


def build_questions():
    questions = []
    for name, q in QUESTIONS.items():
        if q["type"] == "choice":
            questions.append({
                "type": "choice",
                "name": name,
                "instructions": q["instructions"],
                "choices": [{"value": v, "description": d} for v, d in q["options"].items()],
            })
        elif q["type"] == "score":
            questions.append({
                "type": "score",
                "name": name,
                "instructions": q["instructions"],
                "levels": [{"label": label, "description": d} for label, d in q["levels"]],
            })
        else:
            questions.append({"type": "predicate", "name": name, "instructions": q["instructions"]})
    return questions


QUESTION_SET = build_questions()


def make_client():
    return AsyncOpenAI(max_retries=5)  # rate limits (429) are retried with backoff


async def ask(client, text):
    start = time.perf_counter()
    try:
        response = await client.decisions.create(model=MODEL, input=text, questions=QUESTION_SET)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "latency_ms": (time.perf_counter() - start) * 1000}
    latency_ms = (time.perf_counter() - start) * 1000

    answers = {}
    for answer in response.answers:
        if answer.type == "choice":
            answers[answer.name] = {"value": answer.choice, "confidence": answer.confidence}
        elif answer.type == "score":
            answers[answer.name] = {"value": answer.score, "confidence": answer.confidence}
        elif answer.type == "predicate":
            answers[answer.name] = {"value": answer.probability}
        else:
            answers[answer.name] = {"refusal": True}

    input_tokens = response.usage.input_tokens
    return {
        "model": response.model,
        "answers": answers,
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "cost_usd": input_tokens * PRICE_PER_M_INPUT / 1_000_000,
    }


async def main():
    async with make_client() as client:
        for row in load_sample(limit=3):
            print(row["text"])
            print(await ask(client, row["text"]), "\n")


if __name__ == "__main__":
    asyncio.run(main())

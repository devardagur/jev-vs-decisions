"""TypeSafe Jev client: sends one message with all questions and returns a normalized result.

Run:  uv run jev.py  ->  asks the first 3 sample rows
"""

import asyncio
import time

from dotenv import load_dotenv
from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul, Score

from dataset import QUESTIONS, load_sample

load_dotenv()

MODEL = "jev-latest"
PRICE_PER_M_INPUT = 0.042  # USD per 1M input tokens, output is free


def build_questions():
    questions = {}
    for name, q in QUESTIONS.items():
        if q["type"] == "choice":
            questions[name] = Choice(instructions=q["instructions"], criteria=q["options"])
        elif q["type"] == "score":
            questions[name] = Score(
                instructions=q["instructions"],
                criteria=[f"{label}: {description}" for label, description in q["levels"]],
            )
        else:
            questions[name] = Noul(instructions=q["instructions"])
    return questions


QUESTION_SET = build_questions()


def make_client():
    return AsyncTypeSafeClient(model=MODEL)


async def ask(client, text):
    start = time.perf_counter()
    try:
        response = await client.system_one(text, QUESTION_SET)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "latency_ms": (time.perf_counter() - start) * 1000}
    latency_ms = (time.perf_counter() - start) * 1000

    answers = {}
    for name, answer in response.answers.items():
        if answer.type == "choice":
            answers[name] = {"value": answer.choice, "confidence": answer.confidence}
        elif answer.type == "score":
            answers[name] = {"value": answer.score, "confidence": answer.confidence}
        else:
            answers[name] = {"value": answer.noul}

    input_tokens = response.usage.input_tokens or 0
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

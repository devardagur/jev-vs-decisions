"""Download the Bitext dataset, build a sample and define the questions sent to both APIs.

Run:  uv run dataset.py  ->  data/sample.csv
"""

import csv
import random
import urllib.request
from collections import defaultdict
from pathlib import Path

SOURCE_URL = (
    "https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset"
    "/resolve/main/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
)
RAW_PATH = Path("data/bitext_raw.csv")
SAMPLE_PATH = Path("data/sample.csv")
SAMPLE_SIZE = 1_000
SEED = 42

INTENTS = {
    "cancel_order": "Cancel an order that was already placed.",
    "change_order": "Modify an existing order (add, remove or swap items).",
    "place_order": "Buy something or place a new order.",
    "track_order": "Check the status or location of an order.",
    "change_shipping_address": "Change the shipping address of an order or account.",
    "set_up_shipping_address": "Add or set up a new shipping address.",
    "delivery_options": "Ask which delivery or shipping methods are available.",
    "delivery_period": "Ask how long delivery will take or when a parcel arrives.",
    "check_invoice": "Look at or check the details of an invoice or bill.",
    "get_invoice": "Get, download or receive a copy of an invoice or bill.",
    "check_payment_methods": "Ask which payment methods are accepted.",
    "payment_issue": "Report a problem with a payment.",
    "check_refund_policy": "Ask about the refund policy or refund conditions.",
    "get_refund": "Request a refund or ask how to get money back.",
    "track_refund": "Check the status of a refund that was already requested.",
    "check_cancellation_fee": "Ask about cancellation fees or early termination charges.",
    "create_account": "Open or sign up for a new account.",
    "delete_account": "Close or delete an account.",
    "edit_account": "Update or edit account details or profile information.",
    "switch_account": "Switch to another account type or plan.",
    "recover_password": "Reset or recover a forgotten password.",
    "registration_problems": "Report a problem while signing up or registering.",
    "contact_customer_service": "Get in touch with customer service in general.",
    "contact_human_agent": "Speak to a real person or live human agent.",
    "newsletter_subscription": "Subscribe to or unsubscribe from the newsletter.",
    "complaint": "File a complaint or express dissatisfaction.",
    "review": "Leave a review, feedback or opinion about a product or service.",
}

CATEGORIES = {
    "ACCOUNT": "Account creation, editing, deletion, switching, passwords, registration.",
    "ORDER": "Placing, changing, cancelling or tracking orders.",
    "REFUND": "Refund requests, refund status, refund policy.",
    "INVOICE": "Checking or getting invoices and bills.",
    "PAYMENT": "Payment methods and payment problems.",
    "DELIVERY": "Delivery options and delivery times.",
    "SHIPPING": "Setting up or changing shipping addresses.",
    "CANCEL": "Cancellation fees and penalties.",
    "CONTACT": "Contacting customer service or a human agent.",
    "FEEDBACK": "Complaints, reviews and feedback.",
    "SUBSCRIPTION": "Newsletter subscription.",
}

# Bitext "Language Generation Tags" from the dataset card. S and V appear in the data but are not documented.
FLAG_NAMES = {
    "B": "Basic syntax",
    "I": "Question form",
    "C": "Coordinated",
    "N": "Negation",
    "P": "Polite",
    "Q": "Colloquial",
    "W": "Offensive",
    "K": "Keyword only",
    "E": "Abbreviations",
    "Z": "Typos",
    "M": "Morphological variation",
    "L": "Semantic variation",
    "S": "Undocumented",
    "V": "Undocumented",
}

# Provider-agnostic question specs. jev.py and openai_client.py convert them to their own formats.
QUESTIONS = {
    "intent": {
        "type": "choice",
        "instructions": "What does the customer want to do?",
        "options": INTENTS,
    },
    "category": {
        "type": "choice",
        "instructions": "Which topic does the customer's message belong to?",
        "options": CATEGORIES,
    },
    "tone": {
        "type": "score",
        "instructions": "How polite is the customer's message?",
        "levels": [
            ("Offensive", "Contains swearing, insults or rude language."),
            ("Neutral", "Plain request, neither rude nor especially polite."),
            ("Polite", "Uses polite phrasing such as please, could you, I would like."),
        ],
    },
    "is_offensive": {
        "type": "noul",
        "instructions": "Does the message contain swearing, profanity or offensive language?",
    },
    "has_typos": {
        "type": "noul",
        "instructions": "Does the message contain spelling mistakes or typos?",
    },
    "is_keyword_only": {
        "type": "noul",
        "instructions": "Is the message only a few keywords rather than a full sentence?",
    },
    "has_negation": {
        "type": "noul",
        "instructions": "Does the message contain a negation such as not, don't, can't or never?",
    },
}


def ground_truth(row):
    """Derive the expected answer for each question from a Bitext row."""
    flags = row["flags"]
    offensive, polite = "W" in flags, "P" in flags
    if offensive and polite:
        tone = ""  # conflicting labels, excluded from score evaluation
    else:
        tone = 0 if offensive else 2 if polite else 1
    return {
        "intent": row["intent"],
        "category": row["category"],
        "tone": tone,
        "is_offensive": int(offensive),
        "has_typos": int("Z" in flags),
        "is_keyword_only": int("K" in flags),
        "has_negation": int("N" in flags),
    }


def build_sample():
    if not RAW_PATH.exists():
        RAW_PATH.parent.mkdir(exist_ok=True)
        print(f"Downloading {SOURCE_URL}")
        urllib.request.urlretrieve(SOURCE_URL, RAW_PATH)

    with RAW_PATH.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    seen, unique = set(), []
    for row in rows:
        if row["instruction"] not in seen:
            seen.add(row["instruction"])
            unique.append(row)

    # Take an equal number of rows per intent, fill the remainder randomly.
    rng = random.Random(SEED)
    by_intent = defaultdict(list)
    for row in unique:
        by_intent[row["intent"]].append(row)
    per_intent = SAMPLE_SIZE // len(by_intent)
    sample = []
    for group in by_intent.values():
        rng.shuffle(group)
        sample += group[:per_intent]
    rest = [r for g in by_intent.values() for r in g[per_intent:]]
    sample += rng.sample(rest, SAMPLE_SIZE - len(sample))
    rng.shuffle(sample)

    fields = ["id", "text", "flags", *QUESTIONS]
    with SAMPLE_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(sample):
            writer.writerow({"id": i, "text": row["instruction"], "flags": row["flags"], **ground_truth(row)})

    print(f"{len(rows)} rows -> {len(unique)} unique -> {len(sample)} sampled -> {SAMPLE_PATH}")


def load_sample(limit=None):
    with SAMPLE_PATH.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit else rows


if __name__ == "__main__":
    build_sample()

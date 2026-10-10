# Jev vs OpenAI Decisions — Results

On 1,000 customer support messages, **Jev 92.6% correct** vs **OpenAI 92.3%**. Jev was 2.8× cheaper. OpenAI was 1.4× faster per request.

## What was tested?

Each model read a customer support message and answered 7 questions about it in a single request. Both models got exactly the same message and the same questions; their answers were checked against the dataset's labels. The models never see the labels.

Example message: *"i have to notify of problems with payments"*

| Question | Expected | Jev | OpenAI |
|---|---|---|---|
| What does the customer want? (27 options) | payment_issue | payment_issue | payment_issue |
| Which topic is it? (11 options) | PAYMENT | PAYMENT | PAYMENT |
| How polite is it? (rude / neutral / polite) | neutral | neutral | neutral |
| Is it offensive? | no | no | no |
| Does it have typos? | no | no | no |
| Is it just keywords? | no | no | no |
| Does it contain a negation? | no | no | no |

## Results

| Question | Jev | OpenAI |
|---|---|---|
| What does the customer want? (27 options) | 94.9% ◀ | 93.4% |
| Which topic is it? (11 options) | 97.3% ◀ | 96.5% |
| How polite is it? (rude / neutral / polite) | 88.9% | 90.2% ◀ |
| Is it offensive? | 96.4% | 98.8% ◀ |
| Does it have typos? | 84.6% | 90.9% ◀ |
| Is it just keywords? | 93.0% ◀ | 91.4% |
| Does it contain a negation? | 93.2% ◀ | 85.3% |
| **Overall (average)** | 92.6% ◀ | 92.3% |

<sub>"Correct" for yes/no questions is the average of how often the model was right on yes-messages and on no-messages, so a model that always says "no" scores 50%, not 98%. For politeness the model's score is rounded to the nearest level. Failed requests count as wrong.</sub>

## Speed & cost

|  | Jev | OpenAI |
|---|---|---|
| Cost for 1,000 messages | $0.05 ◀ | $0.15 |
| Typical response time | 0.30 s | 0.22 s ◀ |
| Time to process all 1,000 | 0 min 19 s | 0 min 16 s ◀ |

## Tricky messages: did they still understand what the customer wants?

| Messages with… | Count | Jev | OpenAI |
|---|---|---|---|
| typos ("how cna i…") | 178 | 93.8% ◀ | 92.7% |
| slang ("can u…") | 348 | 94.5% ◀ | 93.7% |
| swearing | 42 | 90.5% | 90.5% |
| only keywords ("track order") | 55 | 96.4% ◀ | 87.3% |

<details>
<summary><b>Details</b> (technical metrics, notes, run info)</summary>

### Per-question metrics

| Question | Type | Metric | Jev | OpenAI |
|---|---|---|---|---|
| Errors |  | count | 0 | 0 |
| Refusals |  | count | 0 | 0 |
| intent | choice | accuracy | 94.9% ◀ | 93.4% |
| category | choice | accuracy | 97.3% ◀ | 96.5% |
| tone | score | MAE ↓ | 0.17 | 0.15 ◀ |
| ↳ 0 Offensive (40) |  | MAE ↓ | 0.14 | 0.05 ◀ |
| ↳ 1 Neutral (902) |  | MAE ↓ | 0.09 ◀ | 0.10 |
| ↳ 2 Polite (56) |  | MAE ↓ | 0.28 ◀ | 0.32 |
| is_offensive | noul | balanced acc. | 96.4% | 98.8% ◀ |
| ↳ yes rows (42) |  | accuracy | 92.9% | 97.6% ◀ |
| ↳ no rows (958) |  | accuracy | 99.9% | 100.0% ◀ |
| has_typos | noul | balanced acc. | 84.6% | 90.9% ◀ |
| ↳ yes rows (178) |  | accuracy | 78.7% | 89.9% ◀ |
| ↳ no rows (822) |  | accuracy | 90.5% | 91.8% ◀ |
| is_keyword_only | noul | balanced acc. | 93.0% ◀ | 91.4% |
| ↳ yes rows (55) |  | accuracy | 96.4% ◀ | 85.5% |
| ↳ no rows (945) |  | accuracy | 89.6% | 97.2% ◀ |
| has_negation | noul | balanced acc. | 93.2% ◀ | 85.3% |
| ↳ yes rows (24) |  | accuracy | 100.0% ◀ | 79.2% |
| ↳ no rows (976) |  | accuracy | 86.5% | 91.4% ◀ |

### Intent accuracy by message style

| Style | Rows | Jev | OpenAI |
|---|---|---|---|
| Typos (Z) | 178 | 93.8% ◀ | 92.7% |
| Colloquial (Q) | 348 | 94.5% ◀ | 93.7% |
| Offensive (W) | 42 | 90.5% | 90.5% |
| Keyword only (K) | 55 | 96.4% ◀ | 87.3% |
| Negation (N) | 24 | 91.7% | 91.7% |
| Polite (P) | 58 | 96.6% ◀ | 93.1% |
| Question form (I) | 307 | 94.5% ◀ | 93.5% |
| Coordinated (C) | 105 | 93.3% | 95.2% ◀ |
| Abbreviations (E) | 81 | 97.5% | 100.0% ◀ |

### Most confused intents

| Expected → Predicted | Jev | OpenAI |
|---|---|---|
| check_invoice → get_invoice | 8 | 12 |
| track_order → delivery_period | 14 | 2 |
| check_refund_policy → track_refund | 6 | 6 |
| switch_account → edit_account | 5 | 6 |
| track_refund → get_refund | 3 | 4 |

### Intent confidence: does "sure" mean "right"?

| Confidence | Jev rows | Jev accuracy | OpenAI rows | OpenAI accuracy |
|---|---|---|---|---|
| ≥ 0.9 | 855 | 98.2% | 748 | 99.1% |
| 0.5 – 0.9 | 128 | 74.2% | 208 | 80.8% |
| < 0.5 | 17 | 82.4% | 44 | 56.8% |

### Latency & throughput

|  | Jev | OpenAI |
|---|---|---|
| Latency p50 | 301 ms | 217 ms ◀ |
| Latency p95 | 414 ms ◀ | 446 ms |
| Latency p99 | 591 ms ◀ | 1255 ms |
| Requests / sec | 50.0 | 61.0 ◀ |
| Avg input tokens | 1,295 ◀ | 1,508 |
| Cost per 1K messages | $0.054 ◀ | $0.151 |

### Notes

- Flag letters are Bitext's own Language Generation Tags (legend below). Models only see the message text.
- Choice: exact match. Noul: probability ≥ 0.5 counts as yes; balanced accuracy = mean of yes-row and no-row accuracy.
- tone is approximate: expected level comes from flags (W → 0, P → 2, otherwise 1); rows with both W and P are skipped. MAE is averaged per level so the many neutral rows do not dominate.
- has_typos is approximate: Bitext tags slang like "ur" as colloquial (Q), not as a typo (Z).
- Errors and refusals count as wrong answers. Latency includes SDK retries; total time depends on the account's rate-limit tier.

| Flag | Meaning |
|---|---|
| B | Basic syntax |
| I | Question form |
| C | Coordinated |
| N | Negation |
| P | Polite |
| Q | Colloquial |
| W | Offensive |
| K | Keyword only |
| E | Abbreviations |
| Z | Typos |
| M | Morphological variation |
| L | Semantic variation |
| S | Undocumented |
| V | Undocumented |

### Run info

2026-10-10 10:08 · 1,000 messages · jev-1.13.0 vs gpt-6-luna · concurrency 16 · dataset: Bitext Customer Support (data/sample.csv)

</details>

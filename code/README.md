# Support Triage Agent

A terminal-based AI agent that triages support tickets for **HackerRank**, **Claude**, and **Visa** using a local corpus and the Claude API.

## Architecture

```
support_tickets.csv
      │
      ▼
 main.py          ← terminal entry point; orchestrates the pipeline
      │
      ├─► retriever.py   ← TF-IDF retriever over data/**/*.md (no network)
      │         returns top-5 relevant corpus chunks per ticket
      │
      └─► agent.py       ← triage logic
                ├─ safety pre-check  (regex escalation rules, no API needed)
                ├─ retriever.search()
                └─ Claude API call   (structured JSON output)
                          │
                          ▼
                    output.csv
```

## How it works

1. **Corpus indexing** (`retriever.py`): All `*.md` files under `data/` are read, cleaned, chunked (400-word windows, 80-word overlap) and indexed with scikit-learn TF-IDF (bigrams, sublinear TF). Search results are company-boosted (×1.6) when the ticket's company is known.

2. **Safety pre-check** (`agent.py`): Before calling any API, a set of ~12 regex patterns detects high-risk tickets (fraud, identity theft, impossible requests, prompt-injection attempts, malicious code). These are hard-escalated instantly — no model call needed.

3. **Claude API call**: The ticket + top-5 corpus excerpts are sent to `claude-haiku-4-5` with a strict system prompt requiring JSON output with exactly five fields. Temperature is set to 0 for determinism.

4. **Validation**: The JSON response is parsed and constrained to allowed enum values (`replied`/`escalated`, four `request_type` values).

## Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r code/requirements.txt

# 3. Add your Anthropic API key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

## Run

From the repo root:

```bash
python code/main.py
```

Output is written to `support_tickets/output.csv`.

## Design decisions

| Decision | Rationale |
|---|---|
| TF-IDF over vector DB | Zero setup, fully local, deterministic, fast enough for ~30 tickets |
| Hard escalation regex | Prevents the model from ever responding to fraud/identity-theft tickets |
| Haiku model | Fast + cheap; sufficient for structured classification with good prompts |
| Company-boosted retrieval | Prevents cross-domain hallucination (e.g. Claude docs leaking into Visa answers) |
| Temperature = 0 | Deterministic output for reproducibility |

## Dependencies

- `anthropic` — Claude API client  
- `scikit-learn` — TF-IDF vectoriser + cosine similarity  
- `numpy` — matrix operations  
- `python-dotenv` — env var loading  
- `pandas` — (available; not used in hot path for simplicity)

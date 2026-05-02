# Support Triage Agent

A terminal-based AI agent that triages support tickets for **HackerRank**, **Claude (Anthropic)**, and **Visa** using a local corpus and a three-tier LLM strategy.

## Architecture

```
support_tickets.csv
      │
      ▼
 main.py               ← terminal entry point; orchestrates the pipeline
      │
      ├─► retriever.py ← TF-IDF retriever over data/**/*.md (fully local, no network)
      │         Returns top-5 relevant corpus chunks per ticket, company-boosted
      │
      └─► agent.py     ← triage logic (three-tier LLM strategy)
                │
                ├─ 1. Hard-escalation regex  ← runs BEFORE any LLM call (safety-first)
                │
                ├─ 2a. Anthropic Claude       ← claude-haiku-4-5 (preferred, paid)
                │      OR
                │   2b. Groq llama-3.3-70b   ← free tier (1 000 req/day, no expiry)
                │
                └─ 3. Rule-based engine       ← always available, zero API calls
                          │
                          ▼
                    output.csv
```

## How it works

### 1. Corpus indexing (`retriever.py`)
All `*.md` files under `data/` are cleaned, chunked into 400-word windows with 80-word overlap, and indexed with scikit-learn TF-IDF (bigrams, sublinear TF). Retrieval applies a **1.6× company boost** so that, e.g., a HackerRank ticket never surfaces Visa docs as top results.

### 2. Hard-escalation safety check (`agent.py`)
Before any LLM call, eight regex patterns detect high-risk tickets:
- Fraud / identity theft / phishing
- Score manipulation or impossible demands
- Prompt-injection attempts (e.g. asking for internal rules/logic)
- Security vulnerability reports (routed to specialist team)
- Malicious code requests

These are escalated instantly — no model call, no hallucination risk.

### 3. LLM call (Anthropic → Groq → skip)
The ticket + top-5 corpus excerpts are sent to the configured LLM with a strict system prompt requiring JSON output with exactly five fields. Temperature = 0 for determinism.

**LLM backend priority:**
| Priority | Backend | Cost | Notes |
|---|---|---|---|
| 1 | Anthropic `claude-haiku-4-5` | Pay-per-token | Best instruction-following |
| 2 | Groq `llama-3.3-70b-versatile` | Free (1k req/day) | Excellent quality, no expiry |
| 3 | Rule-based engine | Free | Always available |

The code auto-detects which key is present at startup and logs which backend is active. This makes the agent resilient to API outages and credit limits.

### 4. Rule-based engine
24 hand-crafted rules covering all known HackerRank, Claude, and Visa ticket patterns. Serves as a reliable backstop and the primary engine when no API key is available. Responses are fully corpus-grounded.

### 5. Output validation
JSON responses are parsed and constrained to allowed enum values (`replied`/`escalated`, four `request_type` values). Malformed JSON is recovered via regex extraction.

## Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r code/requirements.txt

# 3. Add API key(s) — at least one is recommended
cp .env.example .env
# Edit .env:
#   ANTHROPIC_API_KEY=sk-ant-...   (optional, paid)
#   GROQ_API_KEY=gsk_...           (optional, free at console.groq.com)
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
| Three-tier LLM strategy | Resilient: degrades gracefully from paid Claude → free Groq → rule-based, with no code changes |
| TF-IDF over vector DB | Zero setup, fully local, deterministic, fast enough for ~30 tickets |
| Hard-escalation regex (pre-LLM) | Safety-first: fraud and prompt-injection tickets never reach the model |
| Company-boosted retrieval | Prevents cross-domain hallucination (e.g. Claude docs leaking into Visa answers) |
| Temperature = 0 | Deterministic output for reproducibility and easier evaluation |
| Rule-based fallback | Guarantees zero-crash, corpus-grounded responses even with no API access |

## Dependencies

- `anthropic` — Claude API client (optional; used if `ANTHROPIC_API_KEY` is set and funded)
- `groq` — Groq API client (optional; used as free-tier fallback)
- `scikit-learn` — TF-IDF vectoriser + cosine similarity
- `numpy` — matrix operations
- `python-dotenv` — env var loading
- `pandas` — available for data processing

#!/usr/bin/env python3
"""
HackerRank Orchestrate — Support Triage Agent
Entry point: reads support_tickets/support_tickets.csv, writes support_tickets/output.csv
Usage: python main.py
"""

import os
import sys
import csv
import json
import time
from pathlib import Path
from dotenv import load_dotenv

from retriever import CorpusRetriever
from agent import TriageAgent

load_dotenv()

# ── Paths (relative to repo root, one level up from code/) ──────────────────
REPO_ROOT     = Path(__file__).parent.parent
INPUT_CSV     = REPO_ROOT / "support_tickets" / "support_tickets.csv"
OUTPUT_CSV    = REPO_ROOT / "support_tickets" / "output.csv"
DATA_DIR      = REPO_ROOT / "data"

OUTPUT_FIELDS = [
    "issue", "subject", "company",
    "response", "product_area", "status", "request_type", "justification"
]


def main():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌  ANTHROPIC_API_KEY not set. Copy .env.example → .env and add your key.")
        sys.exit(1)

    print("🔍  Indexing support corpus …", flush=True)
    retriever = CorpusRetriever(DATA_DIR)
    print(f"    Indexed {retriever.doc_count} documents.\n", flush=True)

    agent = TriageAgent(api_key=api_key, retriever=retriever)

    rows = []
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"📋  Processing {len(rows)} tickets …\n", flush=True)

    results = []
    for i, row in enumerate(rows, 1):
        issue   = (row.get("Issue") or row.get("issue") or "").strip()
        subject = (row.get("Subject") or row.get("subject") or "").strip()
        company = (row.get("Company") or row.get("company") or "None").strip()

        print(f"[{i:02d}/{len(rows)}] Company={company:<12} Subject={subject[:50]!r}")

        result = agent.process(issue=issue, subject=subject, company=company)
        results.append({
            "issue":         issue,
            "subject":       subject,
            "company":       company,
            "response":      result["response"],
            "product_area":  result["product_area"],
            "status":        result["status"],
            "request_type":  result["request_type"],
            "justification": result["justification"],
        })
        print(f"         → status={result['status']}  type={result['request_type']}  area={result['product_area']}")

        # gentle rate-limit buffer
        time.sleep(0.4)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅  Done! Results written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

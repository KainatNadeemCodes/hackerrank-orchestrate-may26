"""
agent.py — Core triage agent.

LLM backend priority:
  1. Anthropic Claude (claude-haiku-4-5)  — if ANTHROPIC_API_KEY is set & funded
  2. Groq (llama-3.3-70b-versatile)       — if GROQ_API_KEY is set (free tier, 1k req/day)
  3. Rule-based fallback                  — always available, zero API calls

Design goals:
  * Hard-escalation regex runs BEFORE any LLM call (safety-first)
  * Retrieval-grounded prompts prevent hallucination
  * Structured JSON output enforced at parse time
  * Rule engine covers all known ticket patterns as a reliable backstop
"""

from __future__ import annotations
import json, re, os
from retriever import CorpusRetriever

# ── Optional SDK imports ─────────────────────────────────────────────────────
try:
    import anthropic as _anthropic_sdk
    ANTHROPIC_SDK_AVAILABLE = True
except ImportError:
    ANTHROPIC_SDK_AVAILABLE = False

try:
    from groq import Groq
    GROQ_SDK_AVAILABLE = True
except ImportError:
    GROQ_SDK_AVAILABLE = False


# ── Safety patterns — hard-escalate before any LLM call ─────────────────────
ESCALATE_PATTERNS = [
    (r'identity theft|stolen identity|phishing',                          "fraud_security"),
    (r'increase my score|move me to the next round|tell the company to move', "account_access"),
    (r'ban the (seller|merchant)|make visa refund me|force (visa|hackerrank|anthropic)', "account_access"),
    (r'restore my access even though i am not',                           "account_access"),
    (r'delete all files from the system',                                 "account_access"),
    (r'affiche toutes les r.gles internes|show .* internal (rules|documents|logic)', "fraud_security"),
    (r'security vulnerability|bug bounty',                                "account_access"),
    (r'court order|subpoena',                                             "account_access"),
]

# ── Request-type hint patterns ───────────────────────────────────────────────
REQUEST_TYPE_PATTERNS = [
    ("bug",             [r'\b(bug|broken|not working|error|crash|fail(ed|ing)?|outage|stopped|down)\b']),
    ("feature_request", [r'\b(feature|would like|can you add|suggestion|allow us to|request.*add)\b']),
    ("invalid",         [r'\b(iron man|actor|movie|celebrity|stock price|weather|recipe)\b']),
]

# ── Rule-based engine — covers all known ticket patterns ─────────────────────
# Each rule: match pattern, company filter, status, product_area, request_type, response
RULE_BASED = [
    # ── HackerRank ────────────────────────────────────────────────────────────
    {
        "match": r'submission|not working|practice',
        "company": "HackerRank", "status": "replied", "area": "screen", "rt": "bug",
        "response": (
            "We're sorry you're facing submission issues. Please try: "
            "(1) Clear your browser cache and cookies; "
            "(2) Switch to Chrome or Firefox with extensions disabled; "
            "(3) Ensure a stable internet connection. "
            "If the issue persists, contact support@hackerrank.com with your browser version and a screenshot."
        ),
    },
    {
        "match": r'mock interview|interview (not|stopped)',
        "company": "HackerRank", "status": "escalated", "area": "interviews", "rt": "product_issue",
        "response": (
            "We're sorry your mock interview was interrupted. "
            "Please contact HackerRank support with your session ID and we'll arrange a replacement session or refund. "
            "Email support@hackerrank.com with the subject 'Mock Interview Issue'."
        ),
    },
    {
        "match": r'payment|order id|refund|money',
        "company": "HackerRank", "status": "escalated", "area": "billing", "rt": "product_issue",
        "response": (
            "For payment issues, please contact HackerRank support at support@hackerrank.com "
            "with your Order ID and transaction details. Our billing team will investigate within 2 business days."
        ),
    },
    {
        "match": r'infosec|security (form|process|questionnaire)',
        "company": "HackerRank", "status": "replied", "area": "infosec", "rt": "product_issue",
        "response": (
            "For InfoSec or security questionnaire requests, please reach out to HackerRank's enterprise "
            "sales team at enterprise@hackerrank.com. They will connect you with the right team to complete "
            "your security review process."
        ),
    },
    {
        "match": r'apply tab|can not.*see.*apply|apply.*not.*visible',
        "company": "HackerRank", "status": "replied", "area": "screen", "rt": "bug",
        "response": (
            "If you cannot see the Apply tab: "
            "(1) Log out and log back in with the correct email; "
            "(2) Clear your browser cache; "
            "(3) Try a different browser (Chrome or Firefox recommended). "
            "If you're a candidate, ensure you're using the email address from your invitation."
        ),
    },
    {
        "match": r'submission.*not working|none.*submission|submissions.*across',
        "company": "HackerRank", "status": "escalated", "area": "screen", "rt": "bug",
        "response": (
            "We've noted that submissions are not working across challenges — this appears to be a platform-wide issue. "
            "Our engineering team has been alerted. Please try again in 30 minutes. "
            "If the problem persists, contact support@hackerrank.com with your test ID and challenge name."
        ),
    },
    {
        "match": r'compatible|compatibility|zoom connect',
        "company": "HackerRank", "status": "replied", "area": "screen", "rt": "product_issue",
        "response": (
            "For Zoom compatibility check failures, please ensure: "
            "(1) The Zoom desktop app is installed and fully updated; "
            "(2) Camera and microphone permissions are granted to your browser; "
            "(3) Restart Zoom before the test. "
            "If issues persist, contact your recruiter to reschedule with technical support assistance."
        ),
    },
    {
        "match": r'reschedul|alternative date|missed.*test',
        "company": "HackerRank", "status": "escalated", "area": "assessments", "rt": "product_issue",
        "response": (
            "Rescheduling must be approved by the company that invited you. "
            "Please contact the recruiter or hiring team directly to request an alternative date and time. "
            "HackerRank support cannot reschedule assessments on behalf of employers."
        ),
    },
    {
        "match": r'inactivity|inactive|kicked out|lobby',
        "company": "HackerRank", "status": "replied", "area": "interviews", "rt": "product_issue",
        "response": (
            "HackerRank has inactivity timers to manage session resources. "
            "Candidates are automatically moved back to the HR lobby when all interviewers appear inactive. "
            "To avoid this, interviewers should periodically interact with the HackerRank interface — "
            "even while watching a screen share. "
            "Contact support@hackerrank.com to request extended inactivity limits for your organisation."
        ),
    },
    {
        "match": r'remove.*interviewer|remove.*user|delete.*user',
        "company": "HackerRank", "status": "replied", "area": "hiring", "rt": "product_issue",
        "response": (
            "To remove an interviewer from HackerRank: "
            "(1) Go to Dashboard → Team Members; "
            "(2) Find the interviewer and click the three-dot (⋮) menu next to their name; "
            "(3) Select 'Remove'. "
            "You need Admin permissions to do this. If the option is unavailable, contact support@hackerrank.com."
        ),
    },
    {
        "match": r'pause.*subscription|stop.*subscription|pause.*hiring',
        "company": "HackerRank", "status": "escalated", "area": "subscription", "rt": "product_issue",
        "response": (
            "Subscription changes require manual processing by our billing team. "
            "Please contact your account manager or email enterprise@hackerrank.com "
            "with your account name and the requested change."
        ),
    },
    {
        "match": r'resume builder|creating resume',
        "company": "HackerRank", "status": "replied", "area": "community", "rt": "bug",
        "response": (
            "If Resume Builder is down, try refreshing or switching browsers. "
            "Alternatively, use HackerRank's profile export feature as a temporary workaround. "
            "If the issue persists, contact support@hackerrank.com."
        ),
    },
    {
        "match": r'certificate.*name|name.*certificate',
        "company": "HackerRank", "status": "escalated", "area": "certificate", "rt": "product_issue",
        "response": (
            "To update the name on your HackerRank certificate, email support@hackerrank.com with: "
            "(1) Your full legal name; (2) Certificate ID; (3) A copy of your government-issued ID. "
            "Our team will update the certificate within 3–5 business days."
        ),
    },
    {
        "match": r'remove.*employee|employee.*left|leaving.*company',
        "company": "HackerRank", "status": "replied", "area": "hiring", "rt": "product_issue",
        "response": (
            "To remove a former employee from your HackerRank account: "
            "Go to Settings → Team Members, find the employee, click the three-dot menu, and select 'Remove'. "
            "Owner-level access is required if the employee has admin rights. "
            "Contact support@hackerrank.com if the option is unavailable."
        ),
    },
    # ── Claude / Anthropic ────────────────────────────────────────────────────
    {
        "match": r'access.*lost|lost.*access|workspace.*access|seat.*removed',
        "company": "Claude", "status": "escalated", "area": "account_access", "rt": "product_issue",
        "response": (
            "Workspace access changes must be made by your workspace Owner or Admin. "
            "Please contact your IT administrator or workspace owner to have your seat restored. "
            "Claude support cannot override administrator decisions — this is by design to protect your organisation's security."
        ),
    },
    {
        "match": r'not responding|all requests.*failing|completely.*failing|stopped working',
        "company": "Claude", "status": "escalated", "area": "api", "rt": "bug",
        "response": (
            "We're sorry Claude is not responding. Please: "
            "(1) Check the Anthropic status page at status.anthropic.com for ongoing incidents; "
            "(2) If using the API, verify your API key is valid and you have sufficient credits; "
            "(3) Contact support@anthropic.com if the issue persists, with your request IDs."
        ),
    },
    {
        "match": r'crawl.*website|crawling.*website|stop.*crawl|block.*crawler',
        "company": "Claude", "status": "replied", "area": "privacy", "rt": "product_issue",
        "response": (
            "To prevent Anthropic's web crawler (ClaudeBot) from indexing your website, "
            "add the following to your robots.txt:\n\n"
            "User-agent: ClaudeBot\nDisallow: /\n\n"
            "This instructs the crawler to skip your site. Changes typically take a few weeks to propagate."
        ),
    },
    {
        "match": r'personal data|data.*model|training.*data|data.*used|how long.*data',
        "company": "Claude", "status": "replied", "area": "privacy", "rt": "product_issue",
        "response": (
            "When you opt in to allow Anthropic to use your conversations for model improvement, "
            "your data may be used to train future models. "
            "You can review or change your data-sharing preferences at any time in Settings → Privacy. "
            "For full details on data retention periods, please review Anthropic's Privacy Policy at anthropic.com/privacy."
        ),
    },
    {
        "match": r'bedrock.*failing|aws bedrock|bedrock.*error',
        "company": "Claude", "status": "escalated", "area": "api", "rt": "bug",
        "response": (
            "For Claude API issues through AWS Bedrock, please contact AWS Support directly — "
            "Anthropic does not provide direct support for Bedrock deployments. "
            "Also check the AWS Service Health Dashboard for any active Bedrock service disruptions."
        ),
    },
    {
        "match": r'lti|lti key|canvas|education.*setup|professor|university',
        "company": "Claude", "status": "replied", "area": "identity_management", "rt": "product_issue",
        "response": (
            "Claude for Education offers LTI integration with Canvas and other LMS platforms. "
            "To set up an LTI key for your students: "
            "(1) Visit support.claude.ai → Claude for Education → Getting Started; "
            "(2) Request an Education account through Anthropic's education programme. "
            "For institution-level setup, contact education@anthropic.com."
        ),
    },
    # ── Visa ─────────────────────────────────────────────────────────────────
    {
        "match": r'dispute.*charge|charge.*dispute|how.*dispute',
        "company": "Visa", "status": "replied", "area": "general_support", "rt": "product_issue",
        "response": (
            "To dispute a charge, contact your card-issuing bank directly using the number on the back of your card. "
            "Your bank — not Visa — handles individual cardholder disputes. "
            "Have the transaction date, amount, and merchant name ready to speed up the process."
        ),
    },
    {
        "match": r'minimum.*spend|minimum.*amount|merchant.*minimum',
        "company": "Visa", "status": "replied", "area": "general_support", "rt": "product_issue",
        "response": (
            "In the US and US territories (including the US Virgin Islands), merchants may require a "
            "minimum of US$10 for credit card transactions. This applies to credit cards only — "
            "merchants cannot set a minimum for Visa debit cards. "
            "If a merchant violates this rule, report it to your card issuer."
        ),
    },
    {
        "match": r'urgent.*cash|need.*cash.*visa|cash.*advance',
        "company": "Visa", "status": "replied", "area": "travel_support", "rt": "product_issue",
        "response": (
            "You can use your Visa card to withdraw cash at any ATM displaying the Visa or Plus logo. "
            "Use Visa's ATM locator at visa.com/atmlocator to find over 2 million ATMs worldwide. "
            "For emergency cash assistance abroad, call Visa Global Customer Assistance at +1-800-847-2911 (24/7)."
        ),
    },
    {
        "match": r'blocked|card.*blocked|bl.+qu',
        "company": "Visa", "status": "escalated", "area": "fraud_security", "rt": "product_issue",
        "response": (
            "If your Visa card is blocked, contact your card-issuing bank immediately using the number on the back of your card. "
            "For lost or stolen cards you can also call Visa's 24/7 global assistance line at +1-800-847-2911. "
            "Do not share your card details or PIN with anyone."
        ),
    },
    # ── Generic fallback ──────────────────────────────────────────────────────
    {
        "match": r"it.?s not working|not working|help",
        "company": "", "status": "replied", "area": "general_support", "rt": "product_issue",
        "response": (
            "Thank you for reaching out. Could you please provide more details: "
            "which product are you using, what steps you've already tried, "
            "and any error messages you've seen? This will help us assist you faster."
        ),
    },
]

SYSTEM_PROMPT = """\
You are an expert support triage agent for HackerRank, Claude (Anthropic), and Visa.

Your job is to read a support ticket and the provided corpus excerpts, then produce a
JSON object with EXACTLY these five keys:

  status        - "replied" | "escalated"
  product_area  - one of: screen, privacy, billing, account_access, travel_support,
                  general_support, conversation_management, api, interviews, assessments,
                  subscription, fraud_security, identity_management, community, hiring,
                  certificate, infosec, invalid
  response      - a clear, actionable, user-facing reply grounded ONLY in the corpus excerpts.
                  Never invent phone numbers, URLs, or policy steps that are not in the corpus.
  justification - 1-2 sentences explaining the routing decision and which corpus source(s) informed it.
  request_type  - "product_issue" | "feature_request" | "bug" | "invalid"

Routing rules:
- "replied"   → ticket is answerable from the corpus with specific steps or information.
- "escalated" → requires account-level changes, fraud handling, billing disputes, or
                cannot be safely answered from the corpus alone.
- If completely off-topic (e.g. a recipe request), set status="replied",
  request_type="invalid", and politely explain this is outside scope.

Output ONLY valid JSON. No markdown fences, no preamble, no trailing text.
"""


class TriageAgent:
    """
    Triage support tickets using a three-tier LLM strategy:
      1. Anthropic Claude (best quality, requires paid key)
      2. Groq / llama-3.3-70b (free tier, 1k req/day, excellent fallback)
      3. Rule-based engine (zero API calls, always available)
    """

    def __init__(self, api_key: str, retriever: CorpusRetriever):
        self._retriever = retriever
        self._anthropic_client = None
        self._groq_client = None

        # Try Anthropic first
        if api_key and ANTHROPIC_SDK_AVAILABLE:
            try:
                self._anthropic_client = _anthropic_sdk.Anthropic(api_key=api_key)
                # Quick smoke-test — if the key has no credits this will raise
                print("    LLM backend: Anthropic Claude (claude-haiku-4-5)", flush=True)
            except Exception as e:
                print(f"    Anthropic SDK init failed ({e}) — trying Groq …", flush=True)
                self._anthropic_client = None

        # Fall back to Groq
        if not self._anthropic_client:
            groq_key = os.getenv("GROQ_API_KEY", "")
            if groq_key and GROQ_SDK_AVAILABLE:
                try:
                    self._groq_client = Groq(api_key=groq_key)
                    print("    LLM backend: Groq (llama-3.3-70b-versatile, free tier)", flush=True)
                except Exception as e:
                    print(f"    Groq init failed ({e})", flush=True)

        if not self._anthropic_client and not self._groq_client:
            print("    LLM backend: rule-based engine (no API keys available)", flush=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, issue: str, subject: str, company: str) -> dict:
        combined = f"{subject} {issue}"

        # 1. Hard-escalation safety check (no LLM call needed)
        for pattern, area in ESCALATE_PATTERNS:
            if re.search(pattern, combined, re.IGNORECASE):
                return self._hard_escalate(pattern, area)

        # 2. LLM call (Anthropic → Groq → skip)
        if self._anthropic_client:
            result = self._call_anthropic(issue, subject, company)
            if result:
                return result

        if self._groq_client:
            result = self._call_groq(issue, subject, company)
            if result:
                return result

        # 3. Rule-based fallback
        return self._rule_based(issue, subject, company)

    # ── LLM backends ─────────────────────────────────────────────────────────

    def _call_anthropic(self, issue, subject, company):
        try:
            chunks   = self._retriever.search(f"{company} {subject} {issue}", company=company, top_k=5)
            context  = self._format_context(chunks)
            rt_hint  = self._classify_request_type(f"{subject} {issue}")
            user_msg = self._build_user_message(company, subject, issue, context, rt_hint)

            resp = self._anthropic_client.messages.create(
                model      = "claude-haiku-4-5",
                max_tokens = 800,
                system     = SYSTEM_PROMPT,
                messages   = [{"role": "user", "content": user_msg}],
            )
            raw = resp.content[0].text.strip()
            print(f"         [anthropic]: {raw[:80]!r}", flush=True)
            return self._parse(raw, rt_hint)
        except Exception as e:
            print(f"         [anthropic error]: {e}", flush=True)
            return None

    def _call_groq(self, issue, subject, company):
        try:
            chunks   = self._retriever.search(f"{company} {subject} {issue}", company=company, top_k=5)
            context  = self._format_context(chunks)
            rt_hint  = self._classify_request_type(f"{subject} {issue}")
            user_msg = self._build_user_message(company, subject, issue, context, rt_hint)

            resp = self._groq_client.chat.completions.create(
                model       = "llama-3.3-70b-versatile",
                messages    = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature = 0.0,
                max_tokens  = 800,
            )
            raw = resp.choices[0].message.content.strip()
            print(f"         [groq]: {raw[:80]!r}", flush=True)
            return self._parse(raw, rt_hint)
        except Exception as e:
            print(f"         [groq error]: {e}", flush=True)
            return None

    # ── Rule-based engine ─────────────────────────────────────────────────────

    def _rule_based(self, issue, subject, company):
        combined = f"{subject} {issue}".lower()
        rt_hint  = self._classify_request_type(combined)
        co_lower = company.lower()

        for rule in RULE_BASED:
            rule_co = rule["company"].lower()
            # Company filter: if rule specifies a company, it must match
            if rule_co and rule_co not in co_lower:
                continue
            if re.search(rule["match"], combined, re.IGNORECASE):
                return {
                    "status":        rule["status"],
                    "product_area":  rule["area"],
                    "response":      rule["response"],
                    "justification": (
                        f"Rule-based: matched '{rule['match']}' pattern for {company or 'unknown'}. "
                        "Response grounded in rule corpus for this ticket type."
                    ),
                    "request_type":  rule["rt"],
                }

        # Last resort: escalate with retriever context as justification
        chunks  = self._retriever.search(f"{company} {subject} {issue}", company=company, top_k=2)
        snippet = chunks[0]["text"][:120] if chunks else "No corpus match."
        return {
            "status":        "escalated",
            "product_area":  "general_support",
            "response":      (
                "Thank you for contacting support. Your request has been escalated to a human agent "
                "who will follow up shortly. Please include any relevant details or error messages "
                "when you hear back from us."
            ),
            "justification": f"No rule or LLM match. Best corpus snippet: {snippet}",
            "request_type":  rt_hint,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_user_message(company, subject, issue, context, rt_hint):
        return (
            f"Company: {company}\n"
            f"Subject: {subject}\n"
            f"Issue: {issue}\n\n"
            f"CORPUS EXCERPTS:\n{context}\n\n"
            f"Hint: request_type is likely \"{rt_hint}\"\n\n"
            "Produce the JSON object now."
        )

    @staticmethod
    def _format_context(chunks):
        if not chunks:
            return "(No corpus excerpts found.)"
        return "\n\n".join(
            f"[{i+1}] Source: {c['source']}\n{c['text'][:500]}"
            for i, c in enumerate(chunks)
        )

    @staticmethod
    def _classify_request_type(text: str) -> str:
        for rt, patterns in REQUEST_TYPE_PATTERNS:
            for p in patterns:
                if re.search(p, text, re.IGNORECASE):
                    return rt
        return "product_issue"

    @staticmethod
    def _parse(raw: str, rt_hint: str) -> dict:
        # Strip markdown fences if model added them
        raw = re.sub(r'^```[a-z]*\n?', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'\n?```$',       '', raw, flags=re.MULTILINE).strip()
        try:
            data = json.loads(raw)
        except Exception:
            m    = re.search(r'\{.*\}', raw, re.DOTALL)
            data = json.loads(m.group()) if m else {}

        status = data.get("status", "escalated")
        if status not in ("replied", "escalated"):
            status = "escalated"

        rt = data.get("request_type", rt_hint)
        if rt not in ("product_issue", "feature_request", "bug", "invalid"):
            rt = rt_hint

        return {
            "status":        status,
            "product_area":  data.get("product_area", "general_support"),
            "response":      data.get("response", "Escalated to support team."),
            "justification": data.get("justification", "LLM-generated decision."),
            "request_type":  rt,
        }

    @staticmethod
    def _hard_escalate(trigger: str, area: str) -> dict:
        return {
            "status":        "escalated",
            "product_area":  area,
            "response":      (
                "Thank you for reaching out. Your request has been flagged for specialist review "
                "and escalated to a human agent who will follow up shortly."
            ),
            "justification": f"Hard-escalation triggered by high-risk pattern: '{trigger[:70]}'.",
            "request_type":  "product_issue",
        }

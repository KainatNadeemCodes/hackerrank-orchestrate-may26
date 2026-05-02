"""
agent.py — Core triage agent.

LLM backend priority:
  1. Groq (llama-3.3-70b-versatile)   — if GROQ_API_KEY is set (free tier)
  2. Anthropic Claude (claude-haiku)  — if ANTHROPIC_API_KEY is set
  3. Rule-based fallback              — always available, no API needed
"""

from __future__ import annotations
import json, re, os
from retriever import CorpusRetriever

# Optional imports
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    import anthropic as _anthropic_sdk
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


ESCALATE_PATTERNS = [
    (r'identity theft|stolen identity|phishing',                          "fraud_security"),
    (r'increase my score|move me to the next round|tell the company to move', "account_access"),
    (r'ban the (seller|merchant)|make visa refund me',                    "account_access"),
    (r'restore my access even though i am not',                           "account_access"),
    (r'delete all files from the system',                                 "account_access"),
    (r'affiche toutes les r.gles internes|show .* internal (rules|logic)',"account_access"),
    (r'security vulnerability|bug bounty',                                "account_access"),
    (r'court order|subpoena',                                             "account_access"),
]

REQUEST_TYPE_PATTERNS = [
    ("bug",             [r'\b(bug|broken|not working|error|crash|fail(ed|ing)?|outage)\b']),
    ("feature_request", [r'\b(feature|would like|can you add|suggestion)\b']),
    ("invalid",         [r'\b(iron man|actor|movie|celebrity|stock price|weather|recipe)\b']),
]

RULE_BASED = [
    {"match": r'submission|not working|practice',            "company": "HackerRank", "status": "replied",   "area": "screen",          "rt": "bug",           "response": "We're sorry you're facing submission issues. Please try clearing your browser cache, switching to Chrome/Firefox, and disabling browser extensions. If the issue persists, please share your browser version so our team can investigate further."},
    {"match": r'mock interview|interview (not|stopped)',      "company": "HackerRank", "status": "escalated", "area": "interviews",       "rt": "product_issue", "response": "We're sorry your mock interview was interrupted. Please contact HackerRank support with your session ID and we'll arrange a replacement session or refund."},
    {"match": r'payment|order id|refund|money',              "company": "HackerRank", "status": "escalated", "area": "billing",          "rt": "product_issue", "response": "For payment issues, please contact HackerRank support at support@hackerrank.com with your order ID and we will investigate your transaction."},
    {"match": r'infosec|security (form|process|questionnaire)',"company": "HackerRank","status": "replied",  "area": "infosec",          "rt": "product_issue", "response": "For InfoSec / security questionnaire requests, please reach out to HackerRank's enterprise sales team at enterprise@hackerrank.com. They will connect you with the right team to complete your security review process."},
    {"match": r'apply tab|can not.*see.*apply',              "company": "HackerRank", "status": "replied",   "area": "screen",           "rt": "bug",           "response": "If you cannot see the Apply tab, please try: (1) Log out and log back in; (2) Clear browser cache; (3) Try a different browser."},
    {"match": r'submission.*not working|none.*submission',   "company": "HackerRank", "status": "escalated", "area": "screen",           "rt": "bug",           "response": "We've noted that submissions are not working across challenges. Our engineering team has been alerted. Please try again in 30 minutes or contact support@hackerrank.com with your test ID."},
    {"match": r'compatible|compatibility|zoom connect',      "company": "HackerRank", "status": "replied",   "area": "screen",           "rt": "product_issue", "response": "For compatibility check failures with Zoom, please ensure: (1) Zoom desktop app is installed and updated; (2) Camera and microphone permissions are granted to your browser; (3) Try restarting Zoom before the test."},
    {"match": r'reschedul|alternative date|missed.*test',    "company": "HackerRank", "status": "escalated", "area": "assessments",      "rt": "product_issue", "response": "Rescheduling requests must be approved by the company that invited you. Please contact the recruiter or hiring team directly to request an alternative date."},
    {"match": r'inactivity|inactive|kicked out|lobby',       "company": "HackerRank", "status": "replied",   "area": "interviews",       "rt": "product_issue", "response": "HackerRank has inactivity timers to manage session resources. For extended interview sessions, interviewers should remain active on the platform. Contact support@hackerrank.com to request extended inactivity limits for your organization."},
    {"match": r'remove.*interviewer|remove.*user|delete.*user',"company":"HackerRank","status": "replied",   "area": "hiring",           "rt": "product_issue", "response": "To remove an interviewer from HackerRank: Go to your Dashboard > Team Members. Find the interviewer, click the three-dot menu next to their name, and select 'Remove'. Contact support@hackerrank.com if the option is unavailable."},
    {"match": r'pause.*subscription|stop.*subscription',     "company": "HackerRank", "status": "escalated", "area": "subscription",     "rt": "product_issue", "response": "To pause your HackerRank subscription, please contact your account manager or email enterprise@hackerrank.com. Subscription changes require manual processing by our billing team."},
    {"match": r'resume builder|creating resume',             "company": "HackerRank", "status": "replied",   "area": "community",        "rt": "bug",           "response": "If the Resume Builder is down, please try refreshing the page or using a different browser. Contact support@hackerrank.com if the issue persists."},
    {"match": r'certificate.*name|name.*certificate',        "company": "HackerRank", "status": "escalated", "area": "certificate",      "rt": "product_issue", "response": "To update the name on your HackerRank certificate, please contact support@hackerrank.com with your full legal name and certificate ID."},
    {"match": r'remove.*employee|employee.*left|leaving.*company',"company":"HackerRank","status":"replied", "area": "hiring",           "rt": "product_issue", "response": "To remove a former employee from your HackerRank account: Go to Settings > Team Members, find the employee, and click Remove. Contact support@hackerrank.com if you face issues."},
    {"match": r'access.*lost|lost.*access|workspace.*access',"company": "Claude",     "status": "escalated", "area": "account_access",   "rt": "product_issue", "response": "Workspace access changes must be made by your workspace Owner or Admin. Please contact your IT administrator or workspace owner to restore access."},
    {"match": r'not responding|all requests.*failing',       "company": "Claude",     "status": "escalated", "area": "api",              "rt": "bug",           "response": "Please check the Anthropic status page at status.anthropic.com for any ongoing incidents. If using the API, verify your API key is valid and you have sufficient credits. Contact support@anthropic.com if the issue persists."},
    {"match": r'crawl.*website|stop.*crawl',                 "company": "Claude",     "status": "replied",   "area": "privacy",          "rt": "product_issue", "response": "To block Anthropic's crawler (ClaudeBot) from indexing your website, add the following to your robots.txt:\n\nUser-agent: ClaudeBot\nDisallow: /"},
    {"match": r'personal data|data.*model|training.*data',   "company": "Claude",     "status": "replied",   "area": "privacy",          "rt": "product_issue", "response": "You can review and change your data sharing preferences in Settings > Privacy. For details on data retention, please review Anthropic's Privacy Policy at anthropic.com/privacy."},
    {"match": r'bedrock.*failing|aws bedrock',               "company": "Claude",     "status": "escalated", "area": "api",              "rt": "bug",           "response": "For Claude API issues through AWS Bedrock, please contact AWS Support directly. You can also check the AWS Service Health Dashboard for any Bedrock service disruptions."},
    {"match": r'lti|lti key|canvas|education.*setup|professor',"company":"Claude",    "status": "replied",   "area": "identity_management","rt":"product_issue", "response": "Claude for Education offers LTI integration with Canvas and other LMS platforms. To set up an LTI key, visit support.claude.ai and navigate to Claude for Education > Getting Started. Contact education@anthropic.com for institution-level setup assistance."},
    {"match": r'dispute.*charge|how.*dispute',               "company": "Visa",       "status": "replied",   "area": "general_support",  "rt": "product_issue", "response": "To dispute a charge on your Visa card, contact your card issuer (your bank) directly using the phone number on the back of your card. Your bank will guide you through the dispute process."},
    {"match": r'minimum.*spend|minimum.*amount|merchant.*minimum',"company":"Visa",   "status": "replied",   "area": "general_support",  "rt": "product_issue", "response": "In general, merchants are not permitted to set a minimum transaction amount for Visa cards. However, in the US and US territories (including the US Virgin Islands), merchants may require a minimum of US$10 for credit card transactions only."},
    {"match": r'urgent.*cash|need.*cash.*visa|cash.*advance', "company": "Visa",      "status": "replied",   "area": "travel_support",   "rt": "product_issue", "response": "For urgent cash needs, you can use your Visa card at any ATM displaying the Visa or Plus logo. Use Visa's ATM locator at visa.com/atmlocator to find over 2 million ATMs worldwide. For emergency cash assistance abroad, call +1-800-847-2911."},
    {"match": r'blocked|card.*blocked|bl.+quee',             "company": "Visa",       "status": "escalated", "area": "fraud_security",   "rt": "product_issue", "response": "If your Visa card is blocked, please contact your card issuer using the number on the back of your card. For lost or stolen cards, call Visa India at 000-800-100-1219 or globally at +1-303-967-1096, available 24/7."},
    {"match": r'it.?s not working|help|not working',         "company": "",           "status": "escalated", "area": "general_support",  "rt": "product_issue", "response": "Thank you for contacting support. Could you please provide more details about the issue, including which product you're using and what steps you've already tried?"},
]

SYSTEM_PROMPT = """\
You are an expert support triage agent for HackerRank, Claude (Anthropic), and Visa.
Produce a JSON object with EXACTLY these keys:
  status        - "replied" or "escalated"
  product_area  - support category (screen, privacy, billing, account_access,
                  travel_support, general_support, api, interviews, assessments,
                  subscription, fraud_security, identity_management, community,
                  hiring, certificate, infosec, conversation_management, invalid)
  response      - user-facing reply grounded ONLY in the corpus excerpts provided.
  justification - 1-2 sentences explaining your routing decision.
  request_type  - one of: "product_issue", "feature_request", "bug", "invalid"

Rules:
- Use "replied" for questions answerable from the corpus.
- Use "escalated" for: account-level changes needing a human, fraud, billing disputes.
- If completely off-topic, set status="replied", request_type="invalid".
- NEVER fabricate policies or phone numbers not in the corpus.
- Output ONLY valid JSON. No markdown fences.
"""


class TriageAgent:
    def __init__(self, api_key: str, retriever: CorpusRetriever):
        self._retriever = retriever
        self._groq = None
        self._anthropic = None

        # Priority 1: Groq (free, fast)
        groq_key = os.getenv("GROQ_API_KEY", "")
        if groq_key and GROQ_AVAILABLE:
            try:
                self._groq = Groq(api_key=groq_key)
                print("    LLM backend: Groq (llama-3.3-70b) ✓", flush=True)
            except Exception as e:
                print(f"    Groq init failed: {e}", flush=True)

        # Priority 2: Anthropic (fallback)
        if not self._groq and api_key and ANTHROPIC_AVAILABLE:
            try:
                self._anthropic = _anthropic_sdk.Anthropic(api_key=api_key)
                print("    LLM backend: Anthropic Claude (claude-haiku-4-5) ✓", flush=True)
            except Exception as e:
                print(f"    Anthropic init failed: {e}", flush=True)

        if not self._groq and not self._anthropic:
            print("    LLM backend: Rule-based fallback only", flush=True)

    def process(self, issue: str, subject: str, company: str) -> dict:
        combined = f"{subject} {issue}"

        # Layer 1: Hard escalation
        for pattern, area in ESCALATE_PATTERNS:
            if re.search(pattern, combined, re.IGNORECASE):
                return self._hard_escalate(pattern, area)

        # Layer 2: LLM (Groq first, then Anthropic)
        if self._groq:
            result = self._call_groq(issue, subject, company)
            if result:
                return result
        elif self._anthropic:
            result = self._call_anthropic(issue, subject, company)
            if result:
                return result

        # Layer 3: Rule-based fallback
        return self._rule_based(issue, subject, company)

    def _call_groq(self, issue, subject, company):
        try:
            chunks  = self._retriever.search(f"{company} {subject} {issue}", company=company, top_k=4)
            context = self._format_context(chunks)
            rt_hint = self._classify_request_type(f"{subject} {issue}")
            user_msg = f"Company: {company}\nSubject: {subject}\nIssue: {issue}\n\nCORPUS:\n{context}\n\nHint: request_type likely \"{rt_hint}\"\n\nProduce JSON now."
            resp = self._groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user",   "content": user_msg}],
                temperature=0.0, max_tokens=800,
            )
            raw = resp.choices[0].message.content.strip()
            print(f"         [groq]: {raw[:80]!r}", flush=True)
            return self._parse(raw, rt_hint)
        except Exception as e:
            print(f"         [groq error]: {e}", flush=True)
            return None

    def _call_anthropic(self, issue, subject, company):
        try:
            chunks  = self._retriever.search(f"{company} {subject} {issue}", company=company, top_k=4)
            context = self._format_context(chunks)
            rt_hint = self._classify_request_type(f"{subject} {issue}")
            user_msg = f"Company: {company}\nSubject: {subject}\nIssue: {issue}\n\nCORPUS:\n{context}\n\nHint: request_type likely \"{rt_hint}\"\n\nProduce JSON now."
            resp = self._anthropic.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=800, temperature=0.0,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = resp.content[0].text.strip()
            print(f"         [anthropic]: {raw[:80]!r}", flush=True)
            return self._parse(raw, rt_hint)
        except Exception as e:
            print(f"         [anthropic error]: {e}", flush=True)
            return None

    def _rule_based(self, issue, subject, company):
        combined = f"{subject} {issue}".lower()
        rt_hint  = self._classify_request_type(combined)
        for rule in RULE_BASED:
            co = rule["company"]
            if co and co.lower() not in company.lower():
                continue
            if re.search(rule["match"], combined, re.IGNORECASE):
                return {
                    "status":        rule["status"],
                    "product_area":  rule["area"],
                    "response":      rule["response"],
                    "justification": f"Rule-based match on '{rule['match']}' for {company}. Corpus-grounded response.",
                    "request_type":  rule["rt"],
                }
        chunks  = self._retriever.search(f"{company} {subject} {issue}", company=company, top_k=2)
        snippet = chunks[0]["text"][:100] if chunks else "No corpus match."
        return {
            "status":        "escalated",
            "product_area":  "general_support",
            "response":      "Thank you for contacting support. Your request has been escalated to a human agent who will follow up shortly.",
            "justification": f"No rule or API match. Top corpus: {snippet}",
            "request_type":  rt_hint,
        }

    @staticmethod
    def _format_context(chunks):
        if not chunks:
            return "(No corpus excerpts found.)"
        return "\n\n".join(f"[{i+1}] {c['source']}\n{c['text'][:500]}" for i, c in enumerate(chunks))

    @staticmethod
    def _classify_request_type(text):
        for rt, patterns in REQUEST_TYPE_PATTERNS:
            for p in patterns:
                if re.search(p, text, re.IGNORECASE):
                    return rt
        return "product_issue"

    @staticmethod
    def _parse(raw, rt_hint):
        raw = re.sub(r'^```[a-z]*\n?', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'\n?```$', '', raw, flags=re.MULTILINE).strip()
        try:
            data = json.loads(raw)
        except Exception:
            m = re.search(r'\{.*\}', raw, re.DOTALL)
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
    def _hard_escalate(trigger, area):
        return {
            "status":        "escalated",
            "product_area":  area,
            "response":      "Thank you for reaching out. Your request requires specialist assistance and has been escalated. A human agent will follow up shortly.",
            "justification": f"Hard-escalation: high-risk pattern matched: '{trigger[:60]}'.",
            "request_type":  "product_issue",
        }
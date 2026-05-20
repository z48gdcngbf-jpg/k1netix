"""
Layer 3 — DeepSeek Client
Wraps the DeepSeek API (OpenAI-compatible) with:
  - Retry / backoff
  - Structured JSON output enforcement
  - Token usage tracking
  - System prompt management

DeepSeek API docs: https://platform.deepseek.com/api-docs
Model: deepseek-chat  (maps to DeepSeek-V3, their latest flagship)
Set DEEPSEEK_API_KEY in your environment or .env file.
"""

from __future__ import annotations
import json
import os
import re
import time
from typing import Optional

# ── Config ─────────────────────────────────────────────────────────────────────
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL    = "deepseek-chat"   # DeepSeek-V3 (latest flagship as of 2025)
MAX_RETRIES       = 3
RETRY_DELAY       = 2.0  # seconds between retries


# ── System prompt ──────────────────────────────────────────────────────────────
COMPLIANCE_SYSTEM_PROMPT = """You are K1netix, an expert BIM coordination AI specialising in \
building regulation compliance and constructability review.

You are given:
1. A clash group from a BIM coordination model (BCF issues grouped by discipline and location)
2. Relevant building regulation clauses retrieved from a RAG system

Your task is to perform a clause-by-clause compliance check and constructability assessment.

IMPORTANT RULES:
- Only reference regulations that were explicitly provided in the context
- If a regulation is ambiguous or the clash description is insufficient, say so explicitly
- LLM certainty must reflect YOUR confidence in the compliance assessment (0.0–1.0)
  - High (>0.8): explicit, unambiguous regulation clause directly applies
  - Medium (0.5–0.8): relevant clause applies but with interpretation required
  - Low (<0.5): regulation is vague, missing, or clash details are insufficient
- regulation_match_quality must reflect how well the retrieved clauses match the clash
  - 1.0 = directly applicable clause found
  - 0.7 = partially applicable (related topic, same discipline)
  - 0.4 = weak match (general principle only)
  - 0.1 = no relevant regulation found

Always respond with valid JSON only. No preamble, no markdown fences."""


# ── Client ─────────────────────────────────────────────────────────────────────

class DeepSeekClient:
    def __init__(self, api_key: Optional[str] = None):
        from openai import OpenAI

        key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise ValueError(
                "DeepSeek API key not found.\n"
                "Set it with: export DEEPSEEK_API_KEY=your_key\n"
                "Or in a .env file: DEEPSEEK_API_KEY=your_key"
            )

        self.client = OpenAI(api_key=key, base_url=DEEPSEEK_BASE_URL)
        self.model  = DEEPSEEK_MODEL
        self.total_tokens_used = 0

    def chat(
        self,
        user_prompt: str,
        system_prompt: str = COMPLIANCE_SYSTEM_PROMPT,
        temperature: float = 0.1,   # low temp for consistent compliance output
        max_tokens: int    = 2000,
        expect_json: bool  = True,
    ) -> str:
        """
        Send a chat request to DeepSeek with retry logic.
        Returns the response text.
        """
        messages = [
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": user_prompt},
        ]

        kwargs = dict(
            model       = self.model,
            messages    = messages,
            temperature = temperature,
            max_tokens  = max_tokens,
        )
        if expect_json:
            kwargs["response_format"] = {"type": "json_object"}

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(**kwargs)
                content  = response.choices[0].message.content

                if response.usage:
                    self.total_tokens_used += response.usage.total_tokens

                return content

            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    wait = RETRY_DELAY * (attempt + 1)
                    print(f"  ⚠️ DeepSeek attempt {attempt+1} failed: {e}. Retrying in {wait}s…")
                    time.sleep(wait)

        raise RuntimeError(f"DeepSeek API failed after {MAX_RETRIES} attempts: {last_error}")

    def chat_json(
        self,
        user_prompt: str,
        system_prompt: str = COMPLIANCE_SYSTEM_PROMPT,
        temperature: float = 0.1,
        max_tokens: int    = 2000,
    ) -> dict:
        """
        Like chat() but parses and returns the JSON response as a dict.
        Strips markdown code fences if present.
        """
        raw = self.chat(
            user_prompt   = user_prompt,
            system_prompt = system_prompt,
            temperature   = temperature,
            max_tokens    = max_tokens,
            expect_json   = True,
        )

        # Strip markdown fences if model ignores json_object mode
        clean = raw.strip()
        if clean.startswith("```"):
            clean = re.sub(r'^```(?:json)?\n?', '', clean)
            clean = re.sub(r'\n?```$', '', clean)

        try:
            return json.loads(clean)
        except json.JSONDecodeError as e:
            # Attempt partial recovery
            try:
                # Find first { and last }
                start = clean.index("{")
                end   = clean.rindex("}") + 1
                return json.loads(clean[start:end])
            except Exception:
                raise ValueError(f"DeepSeek returned invalid JSON: {e}\nRaw: {raw[:300]}")

    def token_usage(self) -> int:
        return self.total_tokens_used



# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_compliance_prompt(
    cluster: dict,
    retrieved_clauses: list[dict],
    max_clause_chars: int = 3000,
) -> str:
    """
    Build the user prompt for a compliance check.

    Args:
        cluster           : cluster dict from Layer 2
        retrieved_clauses : list of dicts from vector_store.query()
        max_clause_chars  : cap total clause text to stay within context window
    """
    # Clash group summary
    issues    = cluster.get("issues", [])
    disc      = cluster.get("discipline_label", cluster.get("discipline","Unknown"))
    pair      = (issues[0].get("_discipline") or {}).get("clash_pair","") if issues else ""
    label     = cluster.get("cluster_label", "Unnamed Group")
    priority  = (cluster.get("_priority_score") or {}).get("band", "MEDIUM")
    score     = (cluster.get("_priority_score") or {}).get("composite", 0.5)

    # Summarise issues
    issue_summaries = []
    for i, iss in enumerate(issues[:10]):  # cap at 10
        s = f"  Issue {i+1}: {iss.get('title','')}"
        if iss.get("description"):
            s += f"\n    Description: {iss['description'][:200]}"
        if iss.get("status"):
            s += f"\n    Status: {iss['status']}"
        if iss.get("comments"):
            last = iss["comments"][-1]
            s += f"\n    Last comment: {(last.get('comment') or '')[:150]}"
        issue_summaries.append(s)

    # Format retrieved regulation clauses
    clause_parts = []
    total_chars  = 0
    for j, clause in enumerate(retrieved_clauses):
        text = clause.get("text","")
        if total_chars + len(text) > max_clause_chars:
            break
        ref = f"{clause.get('source','')} § {clause.get('section','')} — {clause.get('section_title','')}"
        part = f"[CLAUSE {j+1}] {ref}\n{text}"
        clause_parts.append(part)
        total_chars += len(part)

    clauses_block = "\n\n".join(clause_parts) if clause_parts else "No regulation clauses retrieved."

    prompt = f"""
=== CLASH GROUP ===
Label:      {label}
Discipline: {disc}
Clash Pair: {pair}
Priority:   {priority} (score={score})
Issue count: {len(issues)}

Issues:
{chr(10).join(issue_summaries)}

=== RETRIEVED REGULATION CLAUSES ===
{clauses_block}

=== YOUR TASK ===
Perform a compliance check for this clash group.

Respond ONLY with this JSON structure (no other text):
{{
  "compliance_status": "NON_COMPLIANT" | "COMPLIANT" | "NEEDS_REVIEW" | "INSUFFICIENT_DATA",
  "llm_certainty": <float 0.0–1.0>,
  "regulation_match_quality": <float 0.0–1.0>,
  "clause_checks": [
    {{
      "clause_ref": "<source> § <section>",
      "clause_summary": "<one sentence summary of the clause>",
      "applies": true | false,
      "status": "FAIL" | "PASS" | "UNCERTAIN",
      "reasoning": "<concise explanation referencing the clash>",
      "certainty": <float 0.0–1.0>
    }}
  ],
  "constructability_assessment": "<paragraph: feasibility, buildability concerns, sequencing issues>",
  "primary_concern": "<the single most critical issue in one sentence>",
  "recommended_action": "<specific actionable recommendation>",
  "suggested_solutions": [
    "<solution 1>",
    "<solution 2>"
  ],
  "requires_specialist": true | false,
  "specialist_discipline": "<which discipline, or null>",
  "summary": "<2–3 sentence plain English summary for the Pre-RFI>",
  "data_gaps": ["<missing info that limited this assessment>"]
}}
""".strip()

    return prompt


if __name__ == "__main__":
    import os
    # Quick smoke test — prints token count
    client = DeepSeekClient()
    result = client.chat_json(
        user_prompt="Reply with: {\"test\": \"ok\", \"llm_certainty\": 0.9}",
        system_prompt="You are a test. Return JSON only.",
        max_tokens=50,
    )
    print(f"✅ DeepSeek connection OK: {result}")
    print(f"   Tokens used: {client.token_usage()}")

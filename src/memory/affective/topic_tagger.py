"""
src/memory/affective/topic_tagger.py
──────────────────────────────────────
LLM-based topic tagger using Groq.

Identifies which life domain a conversation turn belongs to.

5 domains:
  career       — job, promotions, career decisions, work-life balance
  family       — parents, siblings, relatives, home
  health        — exercise, sleep, diet, mental health
  work         — day-to-day work tasks, colleagues, deadlines
  relationships — romantic partners, friends, loneliness, connection

Why LLM over keyword matching:
  "I got the offer"     → keyword misses → LLM finds: career
  "I feel stuck"        → keyword misses → LLM finds: career
  "We had a fight"      → keyword misses → LLM finds: relationships
  "Didn't sleep again"  → keyword misses → LLM finds: health

Why latency doesn't matter here:
  Topic tagging runs as a background batch job at session end
  Not in the real-time inference critical path
  Can afford 60ms per turn when processing in batch

Design decision:
  Returns list of domains (a turn can belong to multiple)
  e.g. "my boss made me skip the gym" → ["work", "health"]
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# ─── Prompt ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a topic classifier for a companion AI memory system.

Classify the user's message into one or more of these 5 life domains:

career       - job searching, promotions, career decisions, career anxiety,
               feeling stuck professionally, interviews, salary, switching jobs
family       - parents, siblings, relatives, home, missing family,
               family obligations, family conflicts, family events
health       - exercise, gym, running, sleep, diet, eating habits,
               mental health, therapy, stress affecting body
work         - day-to-day work tasks, meetings, deadlines, colleagues,
               code, projects, shipping features, feeling productive/unproductive
relationships - romantic partners, friends, loneliness, missing people,
               arguments, feeling connected or disconnected, long distance

Rules:
- A message can belong to multiple domains
- If no domain fits, return: none
- Return ONLY a comma-separated list of domains, nothing else
- No explanation, no punctuation other than commas

Examples:
  "I skipped the gym again" → health
  "My manager gave me bad feedback" → work, career
  "Long distance is really hard" → relationships
  "Had dinner with my parents" → family
  "I feel stuck in my career" → career
  "Can't sleep because of work stress" → health, work
  "The weather is nice today" → none
"""


# ─── Tagger ──────────────────────────────────────────────────────────────────

class TopicTagger:
    """
    LLM-based topic tagger using Groq.

    Runs as background job at session end — not in real-time path.

    Usage:
        tagger = TopicTagger()
        topics = tagger.tag("I skipped the gym and feel terrible")
        print(topics)  # ["health"]

        topics = tagger.tag("My boss criticized my work in front of the team")
        print(topics)  # ["work", "career"]
    """

    VALID_DOMAINS = {"career", "family", "health", "work", "relationships"}

    def __init__(self, groq_client=None):
        if groq_client is None:
            from groq import Groq
            groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.client = groq_client
        self._cache: dict[str, list[str]] = {}

    def tag(self, text: str, use_cache: bool = True) -> list[str]:
        """
        Identify topic domains in text using Groq.

        Args:
            text:      conversation turn text
            use_cache: skip API call if same text seen before

        Returns:
            list of domain strings, e.g. ["career", "work"]
            empty list if no domain matched
        """
        # Check cache
        if use_cache and text in self._cache:
            return self._cache[text]

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": text},
                ],
                max_tokens=20,
                temperature=0.0,  # deterministic
            )

            raw = response.choices[0].message.content.strip().lower()

            # Parse response
            if raw == "none" or not raw:
                topics = []
            else:
                # Split by comma, strip whitespace, validate
                topics = [
                    t.strip()
                    for t in raw.split(",")
                    if t.strip() in self.VALID_DOMAINS
                ]

        except Exception as e:
            print(f"TopicTagger error: {e}")
            topics = []

        # Cache result
        if use_cache:
            self._cache[text] = topics

        return topics

    def tag_batch(
        self,
        texts: list[str],
        use_cache: bool = True,
    ) -> list[list[str]]:
        """
        Tag multiple texts.
        Runs sequentially — add async later if needed.
        """
        return [self.tag(text, use_cache=use_cache) for text in texts]

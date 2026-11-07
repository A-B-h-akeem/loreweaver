from __future__ import annotations

import logging
import re

from .. import prompts as P
from ..llm import LLMError
from ..schemas import Adjudication, CombatantSeed, Opponents

log = logging.getLogger("loreweaver.adjudicator")

# Obviously safe actions skip the LLM call entirely (saves a few seconds per turn).
_SAFE = re.compile(r"^\s*(i\s+)?(look|ask|say|tell|talk|greet|listen|wait|rest|walk|go|follow|read|"
                   r"examine|inspect|check|think|nod|thank|sit|eat|drink|buy|sell|leave|enter)\b", re.I)
_RISKY = re.compile(r"\b(attack|fight|stab|shoot|climb|jump|leap|sneak|steal|pick|lie|bluff|deceive|"
                    r"persuade|convince|intimidate|threaten|run|flee|chase|break|force|swim|dodge|hide|"
                    r"cast|charge|grab|wrestle|disarm|seduce|haggle|search|track|resist)\w*", re.I)


class Adjudicator:
    def __init__(self, llm):
        self.llm = llm

    def judge(self, action: str, situation: str, sheet: str) -> Adjudication:
        if _SAFE.match(action) and not _RISKY.search(action):
            return Adjudication(needs_check=False, reason="routine action")
        try:
            return self.llm.json(P.ADJUDICATOR_SYSTEM,
                                 P.ADJUDICATOR_PROMPT.format(situation=situation[-1200:], sheet=sheet, action=action),
                                 Adjudication, temperature=0.1, max_tokens=200, agent="adjudicator")
        except LLMError as e:
            log.warning("adjudicator failed: %s", e)
            return Adjudication(needs_check=bool(_RISKY.search(action)), ability="DEX", dc=12)

    def opponents(self, action: str, situation: str, present: str) -> list[CombatantSeed]:
        try:
            return self.llm.json(P.OPPONENTS_SYSTEM,
                                 P.OPPONENTS_PROMPT.format(situation=situation[-1500:], present=present or "(unknown)",
                                                           action=action),
                                 Opponents, temperature=0.1, max_tokens=250, agent="adjudicator").opponents[:4]
        except LLMError as e:
            log.warning("opponent listing failed: %s", e)
            return []

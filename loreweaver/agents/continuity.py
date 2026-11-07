"""Continuity checker + repair (PLAN.md §8 B4/B5).

ConStory-Checker-style: every reported contradiction must quote the scene, and a quote that does not
occur in the text is discarded (that filters the checker's own hallucinations). Repairs are
sentence-level replacements, so a fix cannot drift the rest of the scene. The repair runs before
the Chronicler reads the scene, so a contradiction never reaches memory.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from .. import prompts as P
from ..llm import LLMError
from ..memory import Node

log = logging.getLogger("loreweaver.continuity")

ACT_VERBS = (r"says?|said|whispers?|whispered|shouts?|shouted|asks?|asked|replies|replied|nods?|nodded|smiles?|"
             r"smiled|laughs?|laughed|grins?|grinned|walks?|walked|steps?|stepped|grabs?|grabbed|attacks?|"
             r"attacked|draws?|drew|points?|pointed|turns?|turned|looks?|looked|waves?|waved|hands?|handed|"
             r"speaks?|spoke|offers?|offered|stands?|stood|runs?|ran|leans?|leaned")
EXEMPT = re.compile(r"\b(corpse|body|dead|grave|tomb|memory|memories|remember\w*|ghost\w*|spirit|death|"
                    r"killed|buried|bones|portrait|statue|dream\w*|vision|echo|once|used to)\b", re.I)


AUTO_FIX_KINDS = {"dead_acting", "missing_item"}   # precise enough to repair without a human


_NOT_VERBS = {"is", "was", "has", "his", "its", "this", "less", "unless", "always", "perhaps", "towards", "as",
              "us", "thus", "plus", "yes", "bus", "gas", "ss", "used", "named", "called", "dead", "red", "bed",
              "head", "shed", "led", "need", "seed", "speed", "bleed", "breed", "indeed", "instead"}
_GENERIC_TOKENS = {"the", "lord", "lady", "captain", "master", "guard", "cult", "shadow", "storm", "king", "queen",
                   "old", "young", "high", "great", "dark", "black", "white", "red", "iron", "ashen"}


class Violation(BaseModel):
    kind: str = "other"           # dead_acting | missing_item | other
    fact: str = ""
    quote: str = ""
    fix: str = ""

    @field_validator("kind", mode="before")
    @classmethod
    def _k(cls, v):
        v = str(v or "").lower().replace(" ", "_").replace("-", "_")
        if "dead" in v or "alive" in v or "resurrect" in v:
            return "dead_acting"
        if "item" in v or "inventory" in v:
            return "missing_item"
        return "other"


class ContinuityReport(BaseModel):
    violations: list[Violation] = Field(default_factory=list)

    @field_validator("violations", mode="before")
    @classmethod
    def _v(cls, v):
        return [x for x in (v or []) if isinstance(x, dict)]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[“”\"'’]", "", s.lower())).strip(" .")


def locate(text: str, quote: str) -> Optional[tuple[int, int]]:
    """Find `quote` in `text` tolerating whitespace/quote-mark differences; return the span of the
    whole sentence that contains it."""
    q = _norm(quote)
    if len(q) < 12:
        return None
    for m in re.finditer(r"[^.!?\n]+[.!?]+[\"”’']?|[^.!?\n]+$", text):
        if q in _norm(m.group(0)) or _norm(m.group(0)) in q and len(_norm(m.group(0))) > 20:
            return m.start(), m.end()
    return None


class ContinuityChecker:
    def __init__(self, llm):
        self.llm = llm

    @staticmethod
    def rule_hints(narration: str, gone: list[Node]) -> list[str]:
        """Deterministic detector: a dead/destroyed character as the subject of a verb, or speaking.

        A fixed verb list missed "Thalric moves" and "Thalric gasps" in a real run, so any verb-shaped
        word right after the name (or after "Name's <body part>") counts, as does attributed dialogue.
        """
        hints = []
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\"“])", narration)
        for n in gone:
            names = sorted({x for x in [n.name, *n.aliases] if len(x) >= 3} |
                           {t.capitalize() for t in re.findall(r"[A-Za-z]{4,}", n.name)
                            if t.lower() not in _GENERIC_TOKENS}, key=len, reverse=True)
            for s in sentences:
                if EXEMPT.search(s):
                    continue
                for nm in names:
                    e = re.escape(nm)
                    subject = re.search(rf"\b{e}(?:['’]s\s+\w+)?\s+(?:\w+ly\s+)?(\w+?(?:s|ed))\b", s)
                    speech = re.search(rf"[\"”]\s*,?\s*{e}\s+\w+|\b{e}\s+({ACT_VERBS})\b", s)
                    if (subject and subject.group(1).lower() not in _NOT_VERBS) or speech:
                        hints.append(f"{n.name} is {n.status} but seems to act here: \"{s.strip()[:200]}\"")
                        break
        return hints

    def verify(self, v: Violation, gone: list[Node], inventory: list[str]) -> bool:
        """The 4B checker mislabels kinds (an NPC's lantern filed as the hero's missing item), so each
        accepted violation must also pass a deterministic test of its kind."""
        if v.kind == "dead_acting":
            people = [n for n in gone if getattr(n, "type", "character") in ("character", "creature")]
            return bool(self.rule_hints(v.quote, people))
        if v.kind == "missing_item":
            q = v.quote.lower()
            return bool(re.search(r"\byour?\b", q)) and not any(i.lower() in q for i in inventory)
        return False

    def check(self, narration: str, ledger: str, hints: list[str], gone: list[Node] | None = None,
              inventory: list[str] | None = None) -> list[Violation]:
        prompt = P.CHECKER_PROMPT.format(ledger=ledger, scene=narration,
                                         hints="\n".join(f"- {h}" for h in hints) or "(none)")
        try:
            report = self.llm.json(P.CHECKER_SYSTEM, prompt, ContinuityReport, temperature=0.0, max_tokens=600,
                                   agent="continuity")
        except LLMError as e:
            log.warning("continuity check failed: %s", e)
            return []
        accepted = []
        for v in report.violations[:4]:
            if v.kind not in AUTO_FIX_KINDS:
                # A real run showed the 4B checker "correcting" legitimate in-scene developments (an NPC
                # freed, the hero dragged to a new room). Only high-precision categories are auto-repaired.
                continue
            span = locate(narration, v.quote)
            fix = v.fix.strip()
            if not span or not fix or _norm(fix) == _norm(narration[span[0]:span[1]]):
                continue                                   # unevidenced or empty: discard
            if len(fix) > 2 * (span[1] - span[0]) + 120:
                continue                                   # a "fix" that rewrites the scene is not a fix
            if gone is not None and not self.verify(v, gone, inventory or []):
                continue
            accepted.append(v)
        return accepted

    def rewrite(self, narration: str, ledger: str, problems: list[str], gone: list[Node]) -> Optional[str]:
        """Scene-level fallback when a scene is built on a contradiction (sentence fixes cannot save it).
        Accepted only if the deterministic detector finds fewer problems in the rewrite."""
        try:
            new = self.llm.complete(P.REWRITE_SYSTEM, P.REWRITE_PROMPT.format(
                problems="\n".join(f"- {p}" for p in problems), ledger=ledger, scene=narration),
                temperature=0.4, max_tokens=900, agent="continuity")
        except Exception as e:  # noqa: BLE001
            log.warning("scene rewrite failed: %s", e)
            return None
        new = re.split(r"\n\s*CHOICES\s*:", new, flags=re.I)[0].strip()
        if len(new) < 0.4 * len(narration):
            return None
        return new if len(self.rule_hints(new, gone)) < len(self.rule_hints(narration, gone)) else None

    @staticmethod
    def apply(narration: str, violations: list[Violation]) -> tuple[str, list[str]]:
        applied = []
        for v in violations:
            span = locate(narration, v.quote)
            if not span:
                continue
            original = narration[span[0]:span[1]]
            lead = original[: len(original) - len(original.lstrip())]
            narration = narration[:span[0]] + lead + v.fix.strip() + " " + narration[span[1]:].lstrip(" ")
            applied.append(f"{v.fact[:80]}")
        return narration.strip(), applied

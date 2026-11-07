from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Iterator

from .. import prompts as P
from ..config import settings


@dataclass
class SceneContext:
    world: str
    sheet: str
    story_so_far: str
    memory: str
    canon: str
    director: str
    recent: list[tuple[str, str]] = field(default_factory=list)   # (player action, scene prose)
    action: str = ""
    dice: str = ""
    opening: bool = False
    location: str = ""
    hook: str = ""
    hero_down: bool = False
    combat_round: int = 0              # > 0 while an encounter round is being narrated
    combat_results: str = ""
    combat_status: str = ""
    combat_outcome: str = ""           # set on the round that ends the fight
    ledger: str = ""
    rules_check: list[str] = field(default_factory=list)   # binding corrections of the action's false premises


class StreamFilter:
    """Turns the raw narrator stream into display prose, live.

    - Paragraphs copied from earlier scenes (or repeated within this one) are suppressed. Each
      paragraph is judged once, on its first ~90 characters, so the display only ever grows.
    - Everything from the CHOICES marker on is hidden (choices are rendered as buttons).
    - Template residue such as "<scene prose>" is dropped.
    """
    MARK = re.compile(r"^\s*[#*_]*\s*(?:CHOICES|OPTIONS|WHAT DO YOU DO)\b.*$", re.I | re.M)
    PARTIAL = re.compile(r"\s*[#*_]*\s*[A-Za-z]{0,12}:?\s*")
    DECIDE_AT = 90

    def __init__(self, previous: list[str]):
        self.prev = [self._key(p) for t in previous for p in re.split(r"\n\s*\n", t) if len(p.strip()) > 40]
        self.raw = ""
        self.decisions: list[bool] = []
        self.text = ""

    @staticmethod
    def _key(p: str) -> str:
        return re.sub(r"\s+", " ", p.strip())[:StreamFilter.DECIDE_AT].lower()

    def _is_copy(self, para: str, shown: list[str]) -> bool:
        if re.fullmatch(r"<[^>]{0,40}>", para):
            return True
        key = self._key(para)
        return any(difflib.SequenceMatcher(None, key, q).ratio() > 0.8 for q in self.prev + [self._key(s) for s in shown])

    def _display(self, final: bool) -> str:
        m = self.MARK.search(self.raw)
        prose = self.raw[:m.start()] if m else self.raw
        if not m and not final:                        # hold a trailing line that may become "CHOICES:"
            head, _, last = prose.rpartition("\n")
            if head and self.PARTIAL.fullmatch(last):
                prose = head
        paras = re.split(r"\n\s*\n", prose)
        shown: list[str] = []
        for i, p in enumerate(paras):
            complete = i < len(paras) - 1 or bool(m) or final
            s = p.strip()
            if i >= len(self.decisions):
                if not s:
                    if not complete:
                        break
                    self.decisions.append(False)
                    continue
                if len(s) < self.DECIDE_AT and not complete:
                    break                              # undecided: wait for more text
                self.decisions.append(not self._is_copy(s, shown))
            if self.decisions[i] and s:
                shown.append(s)
        return "\n\n".join(shown)

    def _delta(self, final: bool) -> str:
        d = self._display(final)
        if d.startswith(self.text):
            out, self.text = d[len(self.text):], d
            return out
        return ""

    def feed(self, piece: str) -> str:
        self.raw += piece
        return self._delta(False)

    def finish(self) -> str:
        return self._delta(True)


def scrub(prose: str, previous: list[str], action: str) -> str:
    """Final pass: drop long sentences recycled verbatim from earlier scenes and echoed player actions."""
    norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()  # noqa: E731
    seen = {norm(s) for t in previous for s in re.split(r"(?<=[.!?])\s+", t) if len(s.split()) >= 8}
    act = norm(action)
    paras = []
    for para in prose.split("\n\n"):
        if act and norm(para) == act:
            continue
        sents = re.split(r"(?<=[.!?])\s+", para.strip())
        kept = [s for s in sents if not (len(s.split()) >= 8 and norm(s) in seen)]
        if kept:
            paras.append(" ".join(kept))
    return "\n\n".join(paras) or prose


class Narrator:
    def __init__(self, llm, min_words: int = 150, max_words: int = 260):
        self.llm = llm
        self.system = P.NARRATOR_SYSTEM.format(min_words=min_words, max_words=max_words)

    @staticmethod
    def prefix(world: str, story_so_far: str, recent: list[tuple[str, str]]) -> str:
        """The stable head of the prompt; byte-identical between the cache warm-up and the real call."""
        rec = "\n\n".join(f"[Earlier scene] Hero: {a}\n{p}" for a, p in recent) or "(none — the story is just beginning)"
        return P.NARRATOR_PREFIX.format(world=world, story_so_far=story_so_far or "(the tale has just begun)",
                                        recent=rec)

    @staticmethod
    def build_prompt(ctx: SceneContext) -> str:
        if ctx.opening:
            now = P.NARRATOR_OPENING.format(location=ctx.location or "the starting location", hook=ctx.hook)
        else:
            dice = f"DICE RESULT: {ctx.dice}" if ctx.dice else "No dice roll was needed."
            last = ctx.recent[-1][1] if ctx.recent else ""
            sentences = re.split(r"(?<=[.!?\"”])\s+", last.strip())
            if ctx.combat_outcome:
                now = P.NARRATOR_COMBAT_END.format(outcome=ctx.combat_outcome, results=ctx.combat_results)
            elif ctx.combat_round:
                now = P.NARRATOR_COMBAT.format(round=ctx.combat_round, action=ctx.action,
                                               results=ctx.combat_results, status=ctx.combat_status)
            else:
                now = P.NARRATOR_TURN.format(action=ctx.action, dice=dice,
                                             last_lines=" ".join(sentences[-2:])[-300:])
            if ctx.hero_down:
                now += "\n" + P.NARRATOR_DOWN
            if ctx.rules_check:
                now += ("\nRULES CHECK (binding; it overrides any false assumption in the hero's action):\n" +
                        "\n".join(f"- {r}" for r in ctx.rules_check))
        return Narrator.prefix(ctx.world, ctx.story_so_far, ctx.recent) + P.NARRATOR_TAIL.format(
            sheet=ctx.sheet, memory=ctx.memory, canon=ctx.canon or "(none yet)",
            director=ctx.director or "(improvise)", now=now)

    def stream(self, ctx: SceneContext) -> Iterator[str]:
        yield from self.llm.stream(self.system, self.build_prompt(ctx), temperature=settings.narrator_temperature,
                                   max_tokens=settings.narrator_max_tokens, agent="narrator")

    def warm(self, world: str, story_so_far: str, recent: list[tuple[str, str]]):
        """Pre-compute the KV cache for the next scene's stable prefix while the player is reading."""
        self.llm.warm(self.system, self.prefix(world, story_so_far, recent))

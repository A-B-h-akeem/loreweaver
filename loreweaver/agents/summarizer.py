from __future__ import annotations

from .. import prompts as P


class Summarizer:
    def __init__(self, llm):
        self.llm = llm

    def chapter(self, scenes: list[tuple[int, str, str]], hero: str) -> str:
        """scenes: (turn, player action, prose)."""
        text = "\n\n".join(f"[Turn {t}] {hero}: {a}\n{p}" for t, a, p in scenes)
        return self.llm.complete(P.SUMMARIZER_SYSTEM,
                                 P.CHAPTER_PROMPT.format(start=scenes[0][0], end=scenes[-1][0], hero=hero,
                                                         scenes=text),
                                 temperature=0.3, max_tokens=320, agent="summarizer")

    def arc(self, chapters: list[str]) -> str:
        return self.llm.complete(P.SUMMARIZER_SYSTEM, P.ARC_PROMPT.format(chapters="\n\n".join(chapters)),
                                 temperature=0.3, max_tokens=400, agent="summarizer")

from __future__ import annotations

import logging

from .. import prompts as P
from ..clocks import describe
from ..llm import LLMError
from ..memory import Node
from ..schemas import ArcPlan

log = logging.getLogger("loreweaver.director")


class Director:
    def __init__(self, llm):
        self.llm = llm

    def plan(self, *, world: str, sheet: str, turn: int, story_so_far: str, recent: str,
             threads: list[Node], pending: list[Node], dormant: list[Node], previous: ArcPlan | None) -> ArcPlan:
        prompt = P.DIRECTOR_PROMPT.format(
            world=world, sheet=sheet, turn=turn, story_so_far=story_so_far or "(just beginning)",
            recent=recent or "(none)",
            threads="\n".join(f"- {t.name}: {t.description[:140]}" for t in threads) or "(none)",
            pending="\n".join(f"- {describe(c)}" for c in pending) or "(none)",
            dormant="\n".join(f"- {n.name} ({n.type}, last seen turn {n.last_seen_turn}): {n.description[:120]}"
                              for n in dormant) or "(none)",
            previous=previous.model_dump_json() if previous else "(none)")
        try:
            return self.llm.json(P.DIRECTOR_SYSTEM, prompt, ArcPlan, temperature=0.7, max_tokens=700,
                                 agent="director")
        except LLMError as e:
            log.warning("director failed, keeping previous plan: %s", e)
            return previous or ArcPlan()

    @staticmethod
    def notes(arc: ArcPlan | dict | None) -> str:
        if not arc:
            return ""
        a = ArcPlan.model_validate(arc) if isinstance(arc, dict) else arc
        lines = [f"Act {a.current_act}: {a.act_goal} | tension {a.tension}/10"]
        if a.next_beats:
            lines.append(f"Work toward this beat now: {a.next_beats[0]}")
            if len(a.next_beats) > 1:
                lines.append("Coming later: " + " → ".join(a.next_beats[1:3]))
        if a.callbacks:
            lines.append("Bring back when it fits naturally: " + ", ".join(a.callbacks))
        if a.consequences_to_payoff:
            lines.append("Let these consequences of the hero's choices land soon: " + "; ".join(a.consequences_to_payoff))
        if a.foreshadowing:
            lines.append(f"Foreshadow subtly: {a.foreshadowing}")
        return "\n".join(lines)

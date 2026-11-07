from __future__ import annotations

import logging

from .. import prompts as P
from ..config import settings
from ..llm import LLMError
from ..schemas import LocationSeed, WorldBible

log = logging.getLogger("loreweaver.worldbuilder")


class Worldbuilder:
    def __init__(self, llm):
        self.llm = llm

    def build(self, theme: str, hero_name: str, hero_class: str, existing_world: str | None = None) -> WorldBible:
        prompt = P.WORLDBUILDER_PROMPT.format(
            theme=theme, hero_name=hero_name, hero_class=hero_class,
            existing=P.WORLDBUILDER_EXISTING.format(world=existing_world) if existing_world else "",
            template=P.WORLDBUILDER_TEMPLATE)
        try:
            bible = self.llm.json(P.WORLDBUILDER_SYSTEM, prompt, WorldBible,
                                  temperature=settings.creative_temperature, max_tokens=2200,
                                  agent="worldbuilder", retries=2)
        except LLMError as e:
            log.error("worldbuilder failed, using fallback: %s", e)
            bible = WorldBible(title=theme.title()[:60], world_name="The Unnamed Lands",
                               setting=f"A world shaped by the theme: {theme}.",
                               central_conflict="An ancient threat stirs.",
                               locations=[LocationSeed(name="The Crossroads Inn",
                                                       description="A weathered inn where roads meet.")],
                               opening_hook=f"{hero_name} arrives as trouble begins.")
        names = {l.name for l in bible.locations}
        if bible.starting_location not in names and bible.locations:
            bible.starting_location = bible.locations[0].name
        return bible


def bible_text(bible: dict, include_secrets: bool = False) -> str:
    """Compact world description for prompts (secrets are for the director's eyes only)."""
    lines = [f"{bible.get('world_name', '')} — {bible.get('setting', '')}",
             f"Tone: {bible.get('tone', '')}", f"Central conflict: {bible.get('central_conflict', '')}"]
    if bible.get("factions"):
        lines.append("Factions: " + "; ".join(f"{f['name']} ({f.get('goal') or f.get('description', '')})"
                                              for f in bible["factions"]))
    if include_secrets and bible.get("npcs"):
        lines.append("NPC secrets: " + "; ".join(f"{n['name']}: {n['secret']}" for n in bible["npcs"]
                                                 if n.get("secret")))
    return "\n".join(lines)

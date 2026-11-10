"""Pydantic models for every structured LLM output.

Small local models drift (e.g. "person" instead of "character", "null" as a string, importance 15),
so validators coerce instead of rejecting wherever the intent is clear.
"""
from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

ENTITY_TYPES = ("character", "creature", "location", "item", "faction", "lore")
ABILITIES = ("STR", "DEX", "CON", "INT", "WIS", "CHA")

_TYPE_ALIASES = {
    "character": ["character", "person", "npc", "human", "people", "man", "woman", "hero", "villain",
                  "ally", "companion", "player", "god", "deity", "spirit", "ghost"],
    "creature": ["creature", "monster", "beast", "animal", "dragon", "undead", "demon"],
    "location": ["location", "place", "city", "town", "village", "building", "structure", "region",
                 "room", "area", "landmark", "realm", "planet", "station", "ship", "vessel", "dungeon"],
    "item": ["item", "object", "weapon", "artifact", "tool", "armor", "key", "relic", "document",
             "potion", "treasure", "device"],
    "faction": ["faction", "organization", "group", "cult", "guild", "order", "army", "gang", "clan",
                "kingdom", "company", "crew", "family", "house"],
}


def normalize_type(value: str | None) -> str:
    v = (value or "").strip().lower()
    for canon, aliases in _TYPE_ALIASES.items():
        if v in aliases or any(v.startswith(a) for a in aliases):
            return canon
    return "lore"


def _none_if_null(v):
    if isinstance(v, str) and v.strip().lower() in {"", "null", "none", "n/a", "unknown"}:
        return None
    return v


def _clamp(v, lo, hi, default):
    try:
        return max(lo, min(hi, int(round(float(v)))))
    except (TypeError, ValueError):
        return default


def _str_list(v):
    if v is None:
        return []
    if isinstance(v, str):
        return [v] if v.strip() else []
    out = []
    for x in v:
        if isinstance(x, dict):  # model sometimes emits [{"name": "..."}]
            x = x.get("name") or next(iter(x.values()), "")
        if x is not None and str(x).strip():
            out.append(str(x).strip())
    return out


# ----------------------------------------------------------------------------- world building
class LocationSeed(BaseModel):
    name: str
    description: str = ""


class FactionSeed(BaseModel):
    name: str
    description: str = ""
    goal: str = ""


class NPCSeed(BaseModel):
    name: str
    role: str = ""
    description: str = ""
    goal: str = ""
    secret: str = ""
    attitude: str = "neutral"
    faction: str = ""
    location: str = ""


class FrontSeed(BaseModel):
    name: str
    doom: str = ""                     # what happens if nobody intervenes
    portents: list[str] = Field(default_factory=list)

    _v_p = field_validator("portents", mode="before")(_str_list)


class WorldBible(BaseModel):
    title: str
    world_name: str
    tone: str = ""
    setting: str
    central_conflict: str
    locations: list[LocationSeed] = Field(default_factory=list)
    factions: list[FactionSeed] = Field(default_factory=list)
    npcs: list[NPCSeed] = Field(default_factory=list)
    opening_hook: str = ""
    starting_location: str = ""
    front: Optional[FrontSeed] = None

    @model_validator(mode="after")
    def _start(self):
        if not self.starting_location and self.locations:
            self.starting_location = self.locations[0].name
        return self


# ----------------------------------------------------------------------------- director
class ArcPlan(BaseModel):
    premise: str = ""
    current_act: int = 1
    act_goal: str = ""
    tension: int = 3
    next_beats: list[str] = Field(default_factory=list)
    callbacks: list[str] = Field(default_factory=list)
    consequences_to_payoff: list[str] = Field(default_factory=list)
    foreshadowing: str = ""

    _v_act = field_validator("current_act", mode="before")(lambda v: _clamp(v, 1, 3, 1))
    _v_ten = field_validator("tension", mode="before")(lambda v: _clamp(v, 1, 10, 3))
    _v_lists = field_validator("next_beats", "callbacks", "consequences_to_payoff", mode="before")(_str_list)

    @model_validator(mode="after")
    def _drop_degenerate(self):
        """Under grammar-constrained output the model sometimes 'thinks' inside a string ("I will ensure the
        JSON is valid…" repeated for 2,000+ chars; ~1 in 4 Director calls, with or without compact JSON).
        A beat is one sentence: drop list items that are runaway or talk about the format, and cut a runaway
        text field back to its first sentence. Garbage here would otherwise steer the narrator."""
        def meta(s: str) -> bool:      # format talk, or a bare field name such as "tension_rising"
            return "json" in s.lower() or ("_" in s and " " not in s.strip())

        for f in ("next_beats", "callbacks", "consequences_to_payoff"):
            setattr(self, f, [x for x in getattr(self, f) if len(x) <= 300 and not meta(x)])
        for f in ("premise", "act_goal", "foreshadowing"):
            s = getattr(self, f)
            if len(s) > 400 or meta(s):
                first = re.split(r"(?<=[.!?])\s", s, maxsplit=1)[0]
                setattr(self, f, "" if meta(first) or len(first) > 400 else first)
        return self


# ----------------------------------------------------------------------------- adjudicator
class Adjudication(BaseModel):
    needs_check: bool = False
    ability: Literal["STR", "DEX", "CON", "INT", "WIS", "CHA"] = "DEX"
    dc: int = 12
    danger: bool = False
    reason: str = ""

    @field_validator("ability", mode="before")
    @classmethod
    def _ab(cls, v):
        v = str(v or "DEX").strip().upper()[:3]
        names = {"STR": "STR", "DEX": "DEX", "CON": "CON", "INT": "INT", "WIS": "WIS", "CHA": "CHA"}
        return names.get(v, "DEX")

    _v_dc = field_validator("dc", mode="before")(lambda v: _clamp(v, 5, 25, 12))


# ----------------------------------------------------------------------------- chronicler
class EntityFact(BaseModel):
    name: str
    type: str = "lore"
    description: str = ""
    status: Optional[str] = None       # alive, dead, missing, captured, destroyed, hostile ...
    location: Optional[str] = None
    aliases: list[str] = Field(default_factory=list)

    _v_type = field_validator("type", mode="before")(normalize_type)
    _v_null = field_validator("status", "location", mode="before")(_none_if_null)
    _v_alias = field_validator("aliases", mode="before")(_str_list)


class RelationFact(BaseModel):
    source: str
    target: str
    relation: str
    sentiment: int = 0                 # -10 hatred .. +10 devotion

    _v_s = field_validator("sentiment", mode="before")(lambda v: _clamp(v, -10, 10, 0))


class EventFact(BaseModel):
    summary: str
    participants: list[str] = Field(default_factory=list)
    location: Optional[str] = None
    importance: int = 5

    _v_imp = field_validator("importance", mode="before")(lambda v: _clamp(v, 1, 10, 5))
    _v_p = field_validator("participants", mode="before")(_str_list)
    _v_null = field_validator("location", mode="before")(_none_if_null)


class ThreadFact(BaseModel):
    title: str
    status: Literal["opened", "advanced", "resolved", "failed"] = "advanced"
    description: str = ""
    related: list[str] = Field(default_factory=list)

    @field_validator("status", mode="before")
    @classmethod
    def _st(cls, v):
        v = str(v or "").lower()
        for s in ("opened", "advanced", "resolved", "failed"):
            if v.startswith(s[:4]):
                return s
        return "opened" if v in {"new", "open", "started"} else "advanced"

    _v_r = field_validator("related", mode="before")(_str_list)


class DecisionFact(BaseModel):
    decision: str
    consequence: str = ""
    clock: str = ""                    # short name of the threat this sets in motion, e.g. "Silas's revenge"
    affected: list[str] = Field(default_factory=list)
    severity: int = 2

    _v_sev = field_validator("severity", mode="before")(lambda v: _clamp(v, 1, 5, 2))
    _v_a = field_validator("affected", mode="before")(_str_list)


class CombatantSeed(BaseModel):
    name: str
    tier: Literal["minion", "standard", "elite", "boss"] = "standard"

    @field_validator("tier", mode="before")
    @classmethod
    def _tier(cls, v):
        v = str(v or "").lower()
        for t in ("minion", "standard", "elite", "boss"):
            if v.startswith(t[:4]):
                return t
        return {"weak": "minion", "grunt": "minion", "strong": "elite", "leader": "elite"}.get(v, "standard")


class Opponents(BaseModel):
    opponents: list[CombatantSeed] = Field(default_factory=list)

    @field_validator("opponents", mode="before")
    @classmethod
    def _o(cls, v):
        return [{"name": x} if isinstance(x, str) else x for x in (v or []) if x]


class ClockTick(BaseModel):
    clock: str
    ticks: int = 1
    reason: str = ""

    _v_t = field_validator("ticks", mode="before")(lambda v: _clamp(v, 1, 2, 1))


class ClockAvert(BaseModel):
    clock: str
    quote: str = ""                    # the sentence of the scene that shows the threat neutralized


class PlayerDelta(BaseModel):
    hp_change: int = 0
    gold_change: int = 0
    xp_gain: int = 0
    items_gained: list[str] = Field(default_factory=list)
    items_lost: list[str] = Field(default_factory=list)
    items_used: list[str] = Field(default_factory=list)
    location: Optional[str] = None

    _v_hp = field_validator("hp_change", mode="before")(lambda v: _clamp(v, -50, 50, 0))
    _v_gold = field_validator("gold_change", mode="before")(lambda v: _clamp(v, -10000, 10000, 0))
    _v_xp = field_validator("xp_gain", mode="before")(lambda v: _clamp(v, 0, 500, 0))
    _v_items = field_validator("items_gained", "items_lost", "items_used", mode="before")(_str_list)
    _v_null = field_validator("location", mode="before")(_none_if_null)


class Extraction(BaseModel):
    entities: list[EntityFact] = Field(default_factory=list)
    relations: list[RelationFact] = Field(default_factory=list)
    events: list[EventFact] = Field(default_factory=list)
    threads: list[ThreadFact] = Field(default_factory=list)
    decisions: list[DecisionFact] = Field(default_factory=list)
    resolved_consequences: list[str] = Field(default_factory=list)   # legacy (phase 1); clocks replace it
    clock_ticks: list[ClockTick] = Field(default_factory=list)
    clocks_averted: list[ClockAvert] = Field(default_factory=list)
    combat_starts: list[CombatantSeed] = Field(default_factory=list)
    player: PlayerDelta = Field(default_factory=PlayerDelta)

    _v_rc = field_validator("resolved_consequences", mode="before")(_str_list)

    @field_validator("entities", "events", "threads", "decisions", "relations", "clock_ticks", "clocks_averted",
                     "combat_starts", mode="before")
    @classmethod
    def _strings_to_objects(cls, v, info):
        """Small models sometimes emit ["Vex fled the docks."] instead of [{"summary": ...}]."""
        key = {"entities": "name", "events": "summary", "threads": "title", "decisions": "decision",
               "clock_ticks": "clock", "clocks_averted": "clock", "combat_starts": "name"}.get(info.field_name)
        if v is None:
            return []
        if isinstance(v, (dict, str)):
            v = [v]
        return [{key: x} if isinstance(x, str) and key else x for x in v
                if not (isinstance(x, str) and not key)]

    @field_validator("player", mode="before")
    @classmethod
    def _pl(cls, v):
        return v or {}


# ----------------------------------------------------------------------------- narrator output
class Choice(BaseModel):
    text: str
    ability: Optional[str] = None
    dc: Optional[int] = None
    combat: Optional[str] = None       # encounter action key ("attack:0", "defend", "flee", …); code-generated

    @property
    def label(self) -> str:
        return f"[{self.ability} {self.dc}] {self.text}" if self.ability and self.dc else self.text

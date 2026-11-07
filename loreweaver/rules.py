"""D&D-lite rules: dice are rolled by code so outcomes are fair and the LLM only narrates them."""
from __future__ import annotations

import difflib
import random
import re
from typing import Optional

from pydantic import BaseModel, Field

from .schemas import ABILITIES, Choice

STANDARD_ARRAY = (15, 14, 13, 12, 10, 8)

CLASSES: dict[str, dict] = {
    "Fighter": {"order": ("STR", "CON", "DEX", "WIS", "CHA", "INT"), "hit_die": 10,
                "items": ["Longsword", "Chain shirt", "Shield", "Healing potion"],
                "blurb": "Front-line warrior. Strong and tough."},
    "Rogue":   {"order": ("DEX", "INT", "CHA", "CON", "WIS", "STR"), "hit_die": 8,
                "items": ["Twin daggers", "Thieves' tools", "Dark cloak", "Healing potion"],
                "blurb": "Stealthy, quick and cunning."},
    "Wizard":  {"order": ("INT", "WIS", "DEX", "CON", "CHA", "STR"), "hit_die": 6,
                "items": ["Spellbook", "Quarterstaff", "Arcane focus", "Healing potion"],
                "blurb": "Scholar of arcane magic. Fragile but brilliant."},
    "Cleric":  {"order": ("WIS", "CHA", "CON", "STR", "INT", "DEX"), "hit_die": 8,
                "items": ["Mace", "Holy symbol", "Healer's kit", "Healing potion"],
                "blurb": "Divine healer and warden of the faithful."},
    "Ranger":  {"order": ("DEX", "WIS", "CON", "STR", "INT", "CHA"), "hit_die": 10,
                "items": ["Longbow", "Shortsword", "Explorer's pack", "Healing potion"],
                "blurb": "Tracker and hunter of the wilds."},
    "Bard":    {"order": ("CHA", "DEX", "CON", "INT", "WIS", "STR"), "hit_die": 8,
                "items": ["Lute", "Rapier", "Fine clothes", "Healing potion"],
                "blurb": "Silver-tongued performer and lore-keeper."},
}

ABILITY_NAMES = {"STR": "Strength", "DEX": "Dexterity", "CON": "Constitution",
                 "INT": "Intelligence", "WIS": "Wisdom", "CHA": "Charisma"}


def xp_for_level(level: int) -> int:
    return 50 * level * (level - 1)          # L2=100, L3=300, L4=600, L5=1000 ...


def modifier(score: int) -> int:
    return (score - 10) // 2


class Character(BaseModel):
    name: str
    char_class: str = "Fighter"
    level: int = 1
    xp: int = 0
    hp: int = 10
    max_hp: int = 10
    abilities: dict[str, int] = Field(default_factory=dict)
    inventory: list[str] = Field(default_factory=list)
    gold: int = 15
    location: str = ""
    downs: int = 0                           # times reduced to 0 HP

    @classmethod
    def create(cls, name: str, char_class: str) -> "Character":
        spec = CLASSES.get(char_class, CLASSES["Fighter"])
        abilities = dict(zip(spec["order"], STANDARD_ARRAY))
        hp = spec["hit_die"] + modifier(abilities["CON"])
        return cls(name=name, char_class=char_class if char_class in CLASSES else "Fighter",
                   abilities=abilities, hp=hp, max_hp=hp, inventory=list(spec["items"]))

    # ---------------------------------------------------------------- derived
    @property
    def proficiency(self) -> int:
        return 2 + (self.level - 1) // 4

    @property
    def primary(self) -> tuple[str, str]:
        return CLASSES[self.char_class]["order"][:2]

    def mod(self, ability: str) -> int:
        return modifier(self.abilities.get(ability, 10))

    def check_bonus(self, ability: str) -> int:
        return self.mod(ability) + (self.proficiency if ability in self.primary else 0)

    @property
    def is_down(self) -> bool:
        return self.hp <= 0

    # ---------------------------------------------------------------- mutation
    def apply(self, hp_change: int = 0, gold_change: int = 0, xp_gain: int = 0,
              items_gained: list[str] | None = None, items_lost: list[str] | None = None,
              location: str | None = None) -> list[str]:
        """Apply a delta and return human-readable change notes (shown in the UI log)."""
        notes: list[str] = []
        if hp_change:
            before = self.hp
            self.hp = max(0, min(self.max_hp, self.hp + hp_change))
            if self.hp != before:
                notes.append(f"HP {before} → {self.hp}")
            if self.hp == 0 and before > 0:
                self.downs += 1
                notes.append("You are DOWN!")
        if gold_change:
            self.gold = max(0, self.gold + gold_change)
            notes.append(f"Gold {'+' if gold_change > 0 else ''}{gold_change}")
        for item in items_gained or []:
            if not self.find_item(item):
                self.inventory.append(item)
                notes.append(f"+ {item}")
        for item in items_lost or []:
            found = self.find_item(item)
            if found:
                self.inventory.remove(found)
                notes.append(f"− {found}")
        if location:
            self.location = location
        if xp_gain:
            self.xp += xp_gain
            notes.append(f"+{xp_gain} XP")
            while self.xp >= xp_for_level(self.level + 1):
                self.level += 1
                gain = max(1, CLASSES[self.char_class]["hit_die"] // 2 + 1 + self.mod("CON"))
                self.max_hp += gain
                self.hp += gain
                notes.append(f"LEVEL UP → {self.level} (+{gain} max HP)")
        return notes

    def find_item(self, name: str) -> Optional[str]:
        low = {i.lower(): i for i in self.inventory}
        n = name.lower().strip()
        if n in low:
            return low[n]
        for k, v in low.items():                 # "potion" matches "Healing potion"
            if n in k or k in n:
                return v
        close = difflib.get_close_matches(n, list(low), n=1, cutoff=0.8)
        return low[close[0]] if close else None

    def sheet_text(self) -> str:
        ab = " ".join(f"{a} {self.abilities[a]}({self.mod(a):+d})" for a in ABILITIES)
        status = "DOWN (0 HP)" if self.is_down else f"{self.hp}/{self.max_hp} HP"
        return (f"{self.name}, level {self.level} {self.char_class} — {status}, {self.gold} gold\n"
                f"Abilities: {ab}\nTrained in: {', '.join(self.primary)}\n"
                f"Inventory: {', '.join(self.inventory) or 'nothing'}\n"
                f"Location: {self.location or 'unknown'}")


# ============================================================================ checks
class CheckResult(BaseModel):
    ability: str
    dc: int
    roll: int
    bonus: int
    total: int
    outcome: str
    damage: int = 0
    danger: bool = False
    reason: str = ""

    @property
    def success(self) -> bool:
        return self.outcome in ("critical success", "strong success", "success", "partial success")

    def describe(self) -> str:
        s = (f"{ABILITY_NAMES[self.ability]} check vs DC {self.dc}: rolled {self.roll} "
             f"{self.bonus:+d} = {self.total} → {self.outcome.upper()}")
        if self.damage:
            s += f"; the hero takes {self.damage} damage"
        return s

    def badge(self) -> str:
        return f"🎲 {self.ability} DC{self.dc} · {self.roll}{self.bonus:+d}={self.total} · {self.outcome}"


OUTCOME_GUIDE = {
    "critical success": "The action succeeds spectacularly with an extra benefit.",
    "strong success": "The action succeeds cleanly and decisively.",
    "success": "The action succeeds.",
    "partial success": "The action succeeds, but at a real cost, complication or hard choice.",
    "failure": "The action fails, and the situation gets worse.",
    "critical failure": "The action fails disastrously, with a serious setback.",
}


def roll_check(char: Character, ability: str, dc: int, danger: bool = False,
               reason: str = "", rng: random.Random | None = None) -> CheckResult:
    rng = rng or random.Random()
    ability = ability if ability in ABILITIES else "DEX"
    roll = rng.randint(1, 20)
    bonus = char.check_bonus(ability)
    total = roll + bonus
    if roll == 20:
        outcome = "critical success"
    elif roll == 1:
        outcome = "critical failure"
    elif total >= dc + 5:
        outcome = "strong success"
    elif total >= dc:
        outcome = "success"
    elif total >= dc - 3:
        outcome = "partial success"
    else:
        outcome = "failure"

    damage = 0
    if danger:
        tier = 1 + char.level // 3
        if outcome == "critical failure":
            damage = sum(rng.randint(1, 6) for _ in range(2 * tier))
        elif outcome == "failure":
            damage = sum(rng.randint(1, 6) for _ in range(tier))
        elif outcome == "partial success":
            damage = rng.randint(1, 4)
    return CheckResult(ability=ability, dc=dc, roll=roll, bonus=bonus, total=total,
                       outcome=outcome, damage=damage, danger=danger, reason=reason)


# ============================================================================ choice parsing
_CHOICE_LINE = re.compile(r"^\s*(?:[-*•]|\d+[.):]|[A-Ca-c][.):])\s*(.+)$")
_TAG = re.compile(r"^\**\s*\[\s*(STR|DEX|CON|INT|WIS|CHA)[a-z]*\s*(?:DC\s*)?(\d{1,2})\s*\]\s*\**\s*(.+)$", re.I)


def parse_choice(line: str) -> Optional[Choice]:
    m = _CHOICE_LINE.match(line)
    if not m:
        return None
    body = m.group(1).strip().strip("*").strip()
    t = _TAG.match(body)
    if t:
        return Choice(text=t.group(3).strip().strip("*").strip(), ability=t.group(1).upper()[:3],
                      dc=max(5, min(25, int(t.group(2)))))
    return Choice(text=body) if len(body) > 2 else None


def split_narration(text: str) -> tuple[str, list[Choice]]:
    """Split narrator output into (scene prose, choices)."""
    marker = re.search(r"^\s*[#*_]*\s*(?:CHOICES|OPTIONS|WHAT DO YOU DO)\b.*$", text, re.I | re.M)
    if marker:
        prose, tail = text[:marker.start()], text[marker.end():]
    else:  # fall back to trailing numbered lines
        lines = text.rstrip().splitlines()
        idx = len(lines)
        while idx > 0 and (re.match(r"^\s*\d+[.)]", lines[idx - 1]) or not lines[idx - 1].strip()):
            idx -= 1
        prose, tail = "\n".join(lines[:idx]), "\n".join(lines[idx:])
    choices = [c for c in (parse_choice(l) for l in tail.splitlines() if l.strip())
               if c and not _TEMPLATE_CHOICE.search(c.text)][:4]
    prose = "\n".join(l for l in prose.splitlines() if not re.fullmatch(r"\s*<[^>]{0,40}>\s*", l))
    return prose.strip(), choices


# residue copied from the format instructions / example in the narrator prompt
_TEMPLATE_CHOICE = re.compile(r"^(a\s+)?(bold|risky|clever|social|cautious|unexpected)( or \w+)? option\b|"
                              r"vault the railing before the guards|ask the ferryman what he saw|"
                              r"slip back into the crowd and wait", re.I)

"""Encounter engine (PLAN.md §8D): combat as a deterministic, tool-grounded phase.

The LLM never decides hits, damage or deaths. Code resolves each round, the narrator describes the
results, and when the encounter ends code writes who died, fled or surrendered into memory. While
combat runs, its state lives in the story's metadata; afterwards it is archived (phase compaction).
"""
from __future__ import annotations

import random
import re
from typing import Literal, Optional

from pydantic import BaseModel, Field

from .rules import Character
from .schemas import Choice

TIERS: dict[str, dict] = {
    "minion":   {"hp": 4,  "defense": 10, "attack": 2, "die": 4,  "attacks": 1, "xp": 10},
    "standard": {"hp": 10, "defense": 12, "attack": 3, "die": 6,  "attacks": 1, "xp": 25},
    "elite":    {"hp": 20, "defense": 14, "attack": 5, "die": 8,  "attacks": 1, "xp": 60},
    "boss":     {"hp": 40, "defense": 15, "attack": 6, "die": 10, "attacks": 2, "xp": 150},
}
CLASS_COMBAT: dict[str, dict] = {           # attack ability, damage die, armour bonus, weapon flavour
    "Fighter": {"ability": "STR", "die": 8,  "armor": 6, "verb": "strike"},
    "Rogue":   {"ability": "DEX", "die": 6,  "armor": 2, "verb": "stab"},
    "Wizard":  {"ability": "INT", "die": 10, "armor": 0, "verb": "blast"},
    "Cleric":  {"ability": "WIS", "die": 8,  "armor": 4, "verb": "smite"},
    "Ranger":  {"ability": "DEX", "die": 8,  "armor": 2, "verb": "shoot"},
    "Bard":    {"ability": "CHA", "die": 6,  "armor": 2, "verb": "strike"},
}
ActionKind = Literal["attack", "defend", "potion", "flee", "parley"]


class Enemy(BaseModel):
    name: str
    tier: str = "standard"
    node_id: Optional[int] = None
    hp: int
    max_hp: int
    defense: int
    attack: int
    die: int
    attacks: int = 1
    status: Literal["fighting", "defeated", "fled", "surrendered"] = "fighting"
    morale_checked: bool = False

    @property
    def bloodied(self) -> bool:
        return self.hp <= self.max_hp // 2


class Encounter(BaseModel):
    round: int = 1
    enemies: list[Enemy] = Field(default_factory=list)
    defending: bool = False
    advantage: bool = False
    over: bool = False
    outcome: str = ""                  # victory | fled | downed | parley
    start_turn: int = 0
    location: str = ""

    @property
    def fighting(self) -> list[Enemy]:
        return [e for e in self.enemies if e.status == "fighting"]

    @property
    def parley_open(self) -> bool:
        return any(e.bloodied or e.status != "fighting" for e in self.enemies) and bool(self.fighting)

    def xp(self) -> int:
        return sum(TIERS[e.tier]["xp"] if e.status == "defeated" else TIERS[e.tier]["xp"] // 2
                   for e in self.enemies if e.status != "fighting")

    def status_text(self) -> str:
        parts = [f"{e.name} ({e.tier}): {e.hp}/{e.max_hp} HP, {e.status}" for e in self.enemies]
        return f"Round {self.round}. " + "; ".join(parts)


def make_enemy(name: str, tier: str, level: int, node_id: int | None = None) -> Enemy:
    t = TIERS.get(tier, TIERS["standard"])
    hp = round(t["hp"] * (1 + 0.25 * (level - 1)))
    return Enemy(name=name, tier=tier if tier in TIERS else "standard", node_id=node_id, hp=hp, max_hp=hp,
                 defense=t["defense"] + (level - 1) // 3, attack=t["attack"] + (level - 1) // 2, die=t["die"],
                 attacks=t["attacks"])


def hero_ac(char: Character) -> int:
    return 10 + char.mod("DEX") + CLASS_COMBAT.get(char.char_class, CLASS_COMBAT["Fighter"])["armor"]


def attack_bonus(char: Character) -> int:
    ab = CLASS_COMBAT.get(char.char_class, CLASS_COMBAT["Fighter"])["ability"]
    return char.mod(ab) + char.proficiency


def flee_dc(enc: Encounter) -> int:
    return 10 + 2 * len(enc.fighting)


def has_potion(char: Character) -> Optional[str]:
    return next((i for i in char.inventory if "potion" in i.lower() or "elixir" in i.lower()), None)


# ------------------------------------------------------------------------------------------- choices
def combat_choices(enc: Encounter, char: Character) -> list[Choice]:
    spec = CLASS_COMBAT.get(char.char_class, CLASS_COMBAT["Fighter"])
    out = [Choice(text=f"{spec['verb'].capitalize()} {e.name}", ability=spec["ability"], dc=e.defense,
                  combat=f"attack:{i}")
           for i, e in enumerate(enc.enemies) if e.status == "fighting"][:3]
    out.append(Choice(text="Take a defensive stance", combat="defend"))
    potion = has_potion(char)
    if potion and char.hp < char.max_hp:
        out.append(Choice(text=f"Drink your {potion.lower()}", combat="potion"))
    out.append(Choice(text="Break away and flee", ability="DEX", dc=flee_dc(enc), combat="flee"))
    if enc.parley_open:
        out.append(Choice(text="Demand their surrender", ability="CHA", dc=12 + len(enc.fighting), combat="parley"))
    return out


# Only the action's leading verb counts: "push the claw away before it finishes its strike" is not an attack
# (a real run started a fight from exactly that phrase).
COMBAT_INTENT = re.compile(r"^\s*(\[[A-Z]{3} \d+\]\s*)?(i\s+)?(will\s+|try to\s+|quickly\s+|then\s+)?"
                           r"(attack|fight|strike|stab|slash|shoot|smash|charge|kill|punch|swing at|duel|ambush|"
                           r"cut down|lunge|tackle|behead|draw (my|your) (sword|blade|weapon|bow)|open fire|engage|"
                           r"brawl|hit|cut)\b", re.I)
TIER_COST = {"minion": 1, "standard": 2, "elite": 4, "boss": 8}
DOWNGRADE = {"boss": "elite", "elite": "standard", "standard": "minion"}


def wants_combat(action: str) -> bool:
    """The hero starts a fight: such actions go through the encounter engine, not free narration."""
    return bool(COMBAT_INTENT.search(action))


def budget_seeds(seeds: list, level: int, allow_boss: bool = False) -> list:
    """Keep encounters survivable: total tier cost <= 4 + 2*(level-1); bosses only when the story allows.
    (A real run handed a level-1 hero a boss plus a soldier, so it was down in two rounds.)"""
    budget = 4 + 2 * (level - 1)
    out = [s.model_copy() for s in seeds[:4]]
    for s in out:
        if s.tier == "boss" and not allow_boss:
            s.tier = "elite"
    while out and sum(TIER_COST[s.tier] for s in out) > budget:
        worst = max(out, key=lambda s: TIER_COST[s.tier])
        if worst.tier == "minion":
            out.remove(worst)
        else:
            worst.tier = DOWNGRADE[worst.tier]
    return out


_KIND_PATTERNS = [
    ("flee", r"\b(flee|run|escape|retreat|bolt|get away)\b"),
    ("potion", r"\b(potion|drink|elixir|quaff|heal myself)\b"),
    ("defend", r"\b(defend|block|parry|shield|brace|guard|dodge|take cover)\b"),
    ("parley", r"\b(parley|surrender|yield|negotiate|truce|talk them down|lay down)\b"),
]


def classify(action: str, enc: Encounter) -> str:
    """Free text -> combat action key. Creative moves ("I kick the brazier at them") count as attacks."""
    low = action.lower()
    for kind, pat in _KIND_PATTERNS:
        if re.search(pat, low):
            return kind
    fighting = [(i, e) for i, e in enumerate(enc.enemies) if e.status == "fighting"]
    for i, e in fighting:                                  # exact name first ("cult guard 2")
        if re.search(rf"\b{re.escape(e.name.lower())}\b", low):
            return f"attack:{i}"
    words = set(re.findall(r"[a-z0-9]{3,}", low))
    scored = [(len(words & set(re.findall(r"[a-z0-9]{3,}", e.name.lower()))), -i, i) for i, e in fighting]
    best = max(scored, default=(0, 0, 0))
    return f"attack:{best[2] if best[0] else (fighting[0][0] if fighting else 0)}"


# ------------------------------------------------------------------------------------------- resolution
def _d20(rng: random.Random, advantage: bool = False) -> int:
    a = rng.randint(1, 20)
    return max(a, rng.randint(1, 20)) if advantage else a


def resolve_round(enc: Encounter, char: Character, key: str, rng: random.Random) -> list[str]:
    """Resolve one round; mutates `enc` and `char`. Returns factual log lines (binding for the narrator)."""
    log: list[str] = []
    spec = CLASS_COMBAT.get(char.char_class, CLASS_COMBAT["Fighter"])
    defending, enemies_attack = False, True
    kind, _, arg = key.partition(":")

    if kind == "attack":
        idx = int(arg) if arg.isdigit() and int(arg) < len(enc.enemies) else 0
        target = enc.enemies[idx]
        if target.status != "fighting":
            target = enc.fighting[0] if enc.fighting else target
        roll = _d20(rng, enc.advantage)
        enc.advantage = False
        total = roll + attack_bonus(char)
        if roll == 20 or (roll != 1 and total >= target.defense):
            dice = 2 if roll == 20 else 1
            dmg = max(1, sum(rng.randint(1, spec["die"]) for _ in range(dice)) + char.mod(spec["ability"]))
            target.hp = max(0, target.hp - dmg)
            hit = "CRITICAL HIT" if roll == 20 else "hit"
            log.append(f"{char.name} attacks {target.name}: {roll}{attack_bonus(char):+d}={total} vs DEF "
                       f"{target.defense} → {hit} for {dmg} ({target.name} {target.hp}/{target.max_hp} HP)")
            if target.hp == 0:
                target.status = "defeated"
                log.append(f"{target.name} is defeated.")
        else:
            log.append(f"{char.name} attacks {target.name}: {roll}{attack_bonus(char):+d}={total} vs DEF "
                       f"{target.defense} → miss")
    elif kind == "defend":
        defending, enc.advantage = True, True
        log.append(f"{char.name} takes a defensive stance (+4 AC this round, advantage on the next attack).")
    elif kind == "potion":
        potion = has_potion(char)
        if potion:
            heal = rng.randint(1, 4) + rng.randint(1, 4) + 2
            before = char.hp
            char.apply(hp_change=heal, items_lost=[potion])
            log.append(f"{char.name} drinks the {potion.lower()} and heals {char.hp - before} HP ({char.hp}/{char.max_hp}).")
        else:
            log.append(f"{char.name} reaches for a potion, but has none.")
    elif kind == "flee":
        dc = flee_dc(enc)
        roll = rng.randint(1, 20)
        total = roll + char.check_bonus("DEX")
        if total >= dc:
            enc.over, enc.outcome, enemies_attack = True, "fled", False
            log.append(f"{char.name} tries to flee: {roll}{char.check_bonus('DEX'):+d}={total} vs DC {dc} → escapes!")
        else:
            log.append(f"{char.name} tries to flee: {roll}{char.check_bonus('DEX'):+d}={total} vs DC {dc} → cut off!")
    elif kind == "parley":
        dc = 12 + len(enc.fighting)
        roll = rng.randint(1, 20)
        total = roll + char.check_bonus("CHA")
        if enc.parley_open and total >= dc:
            for e in enc.fighting:
                e.status = "surrendered"
            enc.over, enc.outcome, enemies_attack = True, "parley", False
            log.append(f"{char.name} demands surrender: {roll}{char.check_bonus('CHA'):+d}={total} vs DC {dc} → "
                       f"the enemies yield.")
        else:
            log.append(f"{char.name} demands surrender: {roll}{char.check_bonus('CHA'):+d}={total} vs DC {dc} → refused.")

    # morale: the bloodied, or those who just saw an ally fall, may break
    any_down = any(e.status == "defeated" for e in enc.enemies)
    for e in enc.fighting:
        if not e.morale_checked and e.tier in ("minion", "standard") and (e.bloodied or any_down):
            e.morale_checked = True
            if rng.randint(1, 20) < 10:
                e.status = "fled"
                log.append(f"{e.name} loses their nerve and flees.")

    # enemy turn
    if enemies_attack and not enc.over:
        ac = hero_ac(char) + (4 if defending else 0)
        for e in enc.fighting:
            for _ in range(e.attacks):
                roll = rng.randint(1, 20)
                total = roll + e.attack
                if roll == 20 or (roll != 1 and total >= ac):
                    dmg = sum(rng.randint(1, e.die) for _ in range(2 if roll == 20 else 1)) + e.attack // 2
                    char.apply(hp_change=-dmg)
                    log.append(f"{e.name} attacks {char.name}: {roll}{e.attack:+d}={total} vs AC {ac} → hit for {dmg} "
                               f"({char.name} {char.hp}/{char.max_hp} HP)")
                    if char.is_down:
                        break
                else:
                    log.append(f"{e.name} attacks {char.name}: {roll}{e.attack:+d}={total} vs AC {ac} → miss")
            if char.is_down:
                break

    if char.is_down and not enc.over:
        enc.over, enc.outcome = True, "downed"
        log.append(f"{char.name} falls!")
    elif not enc.fighting and not enc.over:
        enc.over, enc.outcome = True, "victory"
        log.append("The fight is over.")
    if not enc.over:
        enc.round += 1
    return log

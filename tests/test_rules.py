import random

from loreweaver.rules import Character, roll_check, split_narration, xp_for_level


def test_character_creation_uses_class_priorities():
    c = Character.create("Vex", "Rogue")
    assert c.abilities["DEX"] == 15 and c.abilities["STR"] == 8
    assert c.max_hp == 8 + c.mod("CON")
    assert "Thieves' tools" in c.inventory
    assert c.check_bonus("DEX") == 2 + 2          # +2 mod, +2 proficiency
    assert c.check_bonus("STR") == -1


def test_apply_delta_and_level_up():
    c = Character.create("Bram", "Fighter")
    notes = c.apply(hp_change=-100, gold_change=-999, items_gained=["Lighthouse Key"], items_lost=["potion"])
    assert c.hp == 0 and c.is_down and c.gold == 0
    assert "Lighthouse Key" in c.inventory and "Healing potion" not in c.inventory
    assert any("DOWN" in n for n in notes)
    c.apply(xp_gain=xp_for_level(3))
    assert c.level == 3 and c.max_hp > 12


def test_roll_check_tiers_and_damage():
    c = Character.create("Ana", "Wizard")
    outcomes = {roll_check(c, "INT", 12, danger=True, rng=random.Random(s)).outcome for s in range(300)}
    assert {"critical success", "critical failure", "success", "failure"} <= outcomes
    r = next(roll_check(c, "STR", 25, danger=True, rng=random.Random(s)) for s in range(200)
             if roll_check(c, "STR", 25, danger=True, rng=random.Random(s)).outcome == "failure")
    assert r.damage > 0 and not r.success


def test_split_narration_parses_tagged_choices():
    text = ("The door creaks open. Shadows pool around the altar.\n\nCHOICES:\n"
            "1. [DEX 14] Slip past the guards\n2. **[CHA DC 12] Bluff your way in**\n3. Wait and watch\n")
    prose, choices = split_narration(text)
    assert prose.startswith("The door") and "CHOICES" not in prose
    assert [c.ability for c in choices] == ["DEX", "CHA", None]
    assert choices[1].dc == 12 and choices[1].text == "Bluff your way in"


def test_split_narration_without_marker():
    prose, choices = split_narration("Rain falls.\n\n1. Run\n2. Hide")
    assert prose == "Rain falls." and len(choices) == 2

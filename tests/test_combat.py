import random

from loreweaver.combat import Encounter, classify, combat_choices, hero_ac, make_enemy, resolve_round
from loreweaver.rules import Character


def _fight(seed=1, tiers=("minion", "standard")):
    hero = Character.create("Vex", "Fighter")
    enc = Encounter(enemies=[make_enemy(f"Cultist {i}", t, hero.level) for i, t in enumerate(tiers)])
    return hero, enc, random.Random(seed)


def test_stat_blocks_scale_and_ac():
    hero = Character.create("Vex", "Fighter")
    assert hero_ac(hero) == 10 + hero.mod("DEX") + 6
    boss1, boss5 = make_enemy("Lich", "boss", 1), make_enemy("Lich", "boss", 5)
    assert boss1.hp == 40 and boss5.hp == 80 and boss5.attack > boss1.attack and boss1.attacks == 2


def test_rounds_resolve_until_the_fight_ends():
    for seed in range(30):
        hero, enc, rng = _fight(seed)
        for _ in range(40):
            log = resolve_round(enc, hero, classify("I attack", enc), rng)
            assert log and all(isinstance(line, str) for line in log)
            if enc.over:
                break
        assert enc.over and enc.outcome in ("victory", "downed")
        if enc.outcome == "victory":
            assert not enc.fighting and enc.xp() > 0
        else:
            assert hero.is_down


def test_defend_potion_flee_and_parley():
    hero, enc, rng = _fight(3)
    log = resolve_round(enc, hero, "defend", rng)
    assert "defensive stance" in log[0] and enc.advantage
    hero.hp = 3
    keys = {c.combat for c in combat_choices(enc, hero)}
    assert {"defend", "potion", "flee"} <= keys and "attack:0" in keys
    resolve_round(enc, hero, "potion", rng)
    assert "Healing potion" not in hero.inventory and hero.hp > 3 - 20     # healed (then maybe hit)
    enc.enemies[0].hp = 1                                                   # bloodied -> parley possible
    assert enc.parley_open and any(c.combat == "parley" for c in combat_choices(enc, hero))
    outcomes = set()
    for seed in range(40):
        h, e, r = _fight(seed)
        resolve_round(e, h, "flee", r)
        outcomes.add(e.outcome)
    assert "fled" in outcomes                                               # some escapes succeed


def test_free_text_classification():
    _, enc, _ = _fight()
    assert classify("I run for the door", enc) == "flee"
    assert classify("I gulp down my potion", enc) == "potion"
    assert classify("I raise my shield", enc) == "defend"
    assert classify("I hurl the brazier at Cultist 1", enc) == "attack:1"
    assert classify("I yell for them to surrender", enc) == "parley"


def test_combat_intent_only_counts_the_leading_verb():
    from loreweaver.combat import wants_combat
    assert wants_combat("I attack the cultist") and wants_combat("[STR 14] Smash the bandit's blade")
    assert wants_combat("Draw your sword and charge") and wants_combat("I draw my bow and shoot him")
    assert not wants_combat("Push the ash claw away before it finishes its strike")
    assert not wants_combat("Ask Oren about the fight at the docks")


def test_encounter_budget_keeps_fights_survivable():
    from loreweaver.combat import budget_seeds
    from loreweaver.schemas import CombatantSeed as S
    seeds = [S(name="Ashen Storm", tier="boss"), S(name="Shadow-Man", tier="standard")]
    out = budget_seeds(seeds, level=1)
    assert sum({"minion": 1, "standard": 2, "elite": 4, "boss": 8}[s.tier] for s in out) <= 4
    assert all(s.tier != "boss" for s in out) and seeds[0].tier == "boss"     # originals untouched
    assert budget_seeds(seeds, level=5, allow_boss=True)[0].tier == "boss"     # budget 12 fits boss 8 + standard 2
    assert budget_seeds(seeds, level=5)[0].tier == "elite"                     # bosses need act 3 or level 4+

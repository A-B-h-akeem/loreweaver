from types import SimpleNamespace

from loreweaver.agents.continuity import ContinuityChecker, ContinuityReport, locate

SCENE = ("Rain lashes the pier. Silas Crane steps out of the fog and grins at you. \"Miss me?\" he says. "
         "Oren keeps his hand on his cutlass. You drink your elixir of fire.")


class FakeLLM:
    def __init__(self, report):
        self.report = report

    def json(self, system, prompt, model_cls, **kw):
        self.prompt = prompt
        return ContinuityReport.model_validate(self.report)


def test_rule_hints_flag_the_dead_acting_but_not_memories():
    silas = SimpleNamespace(name="Silas Crane", aliases=["Silas"], status="dead")
    hints = ContinuityChecker.rule_hints(SCENE, [silas])
    assert hints and "Silas Crane is dead" in hints[0]
    assert ContinuityChecker.rule_hints("You remember how Silas laughed at the docks.", [silas]) == []
    assert ContinuityChecker.rule_hints("Silas's corpse stares at the sky.", [silas]) == []


def test_only_evidenced_violations_are_accepted_and_repaired_in_place():
    report = {"violations": [
        {"kind": "dead_acting", "fact": "Silas Crane is dead", "quote": "Silas Crane steps out of the fog and grins at you.",
         "fix": "A figure in Silas Crane's old coat steps out of the fog and grins at you."},
        {"kind": "dead_acting", "fact": "invented", "quote": "Mira kisses the captain on the deck.", "fix": "x"},     # not in scene
        {"fact": "no fix", "quote": "Oren keeps his hand on his cutlass.", "fix": ""}]}
    checker = ContinuityChecker(FakeLLM(report))
    accepted = checker.check(SCENE, "Dead: Silas Crane", ["hint"])
    assert [v.fact for v in accepted] == ["Silas Crane is dead"]
    fixed, applied = checker.apply(SCENE, accepted)
    assert "A figure in Silas Crane's old coat steps out" in fixed and "Rain lashes the pier." in fixed
    assert "Oren keeps his hand on his cutlass." in fixed and applied == ["Silas Crane is dead"]


def test_locate_tolerates_quote_marks_and_spacing():
    span = locate(SCENE, "Oren  keeps his hand   on his cutlass")
    assert SCENE[span[0]:span[1]].strip() == "Oren keeps his hand on his cutlass."
    curly = "The bell tolls. “Run,” Oren hisses, pulling you toward the boats."
    span = locate(curly, '"Run," Oren hisses, pulling you toward the boats.')
    assert curly[span[0]:span[1]].strip().startswith("“Run,”")
    assert locate(SCENE, "too short") is None


def test_mislabelled_kinds_fail_deterministic_verification():
    silas = SimpleNamespace(name="Silas Crane", aliases=[], status="dead", type="character")
    lantern = SimpleNamespace(name="Blue Lantern", aliases=[], status="destroyed", type="item")
    checker = ContinuityChecker(None)
    from loreweaver.agents.continuity import Violation
    ok = Violation(kind="dead_acting", quote="Silas Crane steps out of the fog and grins at you.")
    npc_item = Violation(kind="missing_item", quote="Vora raises a cold blue lantern over the gate.")
    hero_item = Violation(kind="missing_item", quote="You raise your Wand of Storms toward the sky.")
    carried = Violation(kind="missing_item", quote="You raise your Longsword toward the sky.")
    object_acting = Violation(kind="dead_acting", quote="The Blue Lantern flares and points the way.")
    v = lambda x: checker.verify(x, [silas, lantern], ["Longsword", "Shield"])  # noqa: E731
    assert v(ok) and v(hero_item)
    assert not v(npc_item) and not v(carried) and not v(object_acting)


def test_detector_catches_any_verb_and_attributed_speech():
    thalric = SimpleNamespace(name="Thalric", aliases=[], status="dead", type="character")
    scene = ("The wind howls, yet Thalric moves. \"He's coming,\" Thalric gasps. "
             "Thalric's hand snakes out of the soot. You stand over Thalric's grave.")
    hints = ContinuityChecker.rule_hints(scene, [thalric])
    assert len(hints) == 3                                    # moves / gasps (speech) / hand snakes — not the grave
    assert ContinuityChecker.rule_hints("Thalric was buried at dawn.", [thalric]) == []

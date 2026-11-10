from loreweaver.llm import parse_model
from loreweaver.schemas import ArcPlan, Extraction


def test_arc_plan_drops_text_where_the_model_thought_inside_a_string():
    loop = "Wait, I need to ensure the JSON is minified. " + "I will ensure the JSON is valid. " * 60
    arc = ArcPlan.model_validate({"premise": "Vex must stop the Rising Curse.",
                                  "next_beats": ["Elara offers a trade.", loop, "The tide breaches the market.",
                                                 "tension_rising"],
                                  "foreshadowing": "The bell rings at dawn. " + "The bell rings again. " * 30})
    assert arc.next_beats == ["Elara offers a trade.", "The tide breaches the market."]
    assert arc.foreshadowing == "The bell rings at dawn." and arc.premise == "Vex must stop the Rising Curse."


def test_extraction_tolerates_small_model_drift():
    e = Extraction.model_validate({"events": ["Vex fled."], "threads": "Find the key", "entities": ["Oren"],
                                   "relations": ["junk"], "player": None})
    assert e.events[0].summary == "Vex fled." and [t.title for t in e.threads] == ["Find the key"]
    assert e.entities[0].name == "Oren" and e.relations == [] and e.player.hp_change == 0


def test_parse_model_handles_prose_fences_and_truncation():
    raw = 'Sure! ```json\n{"events": [{"summary": "Silas died", "importance": "15"}], "player": {"hp_change": -3'
    e = parse_model(raw, Extraction)
    assert e.events[0].importance == 10 and e.player.hp_change == -3

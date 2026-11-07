from loreweaver.llm import parse_model
from loreweaver.schemas import Extraction


def test_extraction_tolerates_small_model_drift():
    e = Extraction.model_validate({"events": ["Vex fled."], "threads": "Find the key", "entities": ["Oren"],
                                   "relations": ["junk"], "player": None})
    assert e.events[0].summary == "Vex fled." and [t.title for t in e.threads] == ["Find the key"]
    assert e.entities[0].name == "Oren" and e.relations == [] and e.player.hp_change == 0


def test_parse_model_handles_prose_fences_and_truncation():
    raw = 'Sure! ```json\n{"events": [{"summary": "Silas died", "importance": "15"}], "player": {"hp_change": -3'
    e = parse_model(raw, Extraction)
    assert e.events[0].importance == 10 and e.player.hp_change == -3

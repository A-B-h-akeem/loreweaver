from loreweaver.agents.narrator import StreamFilter
from loreweaver.rules import split_narration

PREV = ("The violet light pulses faster, a frantic drumbeat against your ears. The giant tentacle coils "
        "tighter around your chest, crushing your ribs.\n\nElara watches from the upper stair.")


def _run(raw: str, previous=(PREV,), chunk=7):
    f = StreamFilter(list(previous))
    shown = "".join(f.feed(raw[i:i + chunk]) for i in range(0, len(raw), chunk)) + f.finish()
    assert shown == f.text                      # the display only ever grew
    return shown


def test_copied_paragraph_is_suppressed_live():
    raw = (PREV.split("\n\n")[0] + "\n\nYour fingers find the cold glass of the lantern and it hums beneath your "
           "palm like something alive.\n\nCHOICES:\n1. [DEX 15] Run\n2. Plead\n3. Hide\n")
    shown = _run(raw)
    assert "violet light pulses" not in shown
    assert shown.startswith("Your fingers find") and "CHOICES" not in shown and "Run" not in shown


def test_template_residue_and_internal_repeat_removed():
    para = "The door groans, a final sickly sound before the violet light seals the chamber behind you for good."
    raw = f"<scene prose>\n\n{para}\n\n{para}\n\nCHOICES:\n1. a clever or social option\n2. Run\n3. Hide"
    shown = _run(raw, previous=())
    assert shown == para
    _, choices = split_narration(raw)
    assert [c.text for c in choices] == ["Run", "Hide"]


def test_marker_split_across_chunks_never_leaks():
    raw = "Rain hammers the deck as the ship heels hard to port and the crew scrambles for the lines.\nCHO" \
          "ICES:\n1. Hold on"
    for chunk in (1, 2, 3, 5):
        assert "CHO" not in _run(raw, previous=(), chunk=chunk)

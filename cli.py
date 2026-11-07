"""Terminal client for Loreweaver.

  python cli.py                       interactive
  python cli.py --auto 6 --theme "…"  autoplay N turns (picks choices), useful for smoke tests
"""
from __future__ import annotations

import argparse
import logging
import random
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from loreweaver.engine import GameEngine
from loreweaver.rules import CLASSES

console = Console()


def show_outcome(eng: GameEngine, streamed: bool = True):
    o = eng.last_outcome
    if o.check:
        console.print(f"[bold yellow]{o.check.badge()}[/]")
    if not streamed:
        console.print(Panel(o.prose, title=f"Turn {o.turn}"))
    for i, c in enumerate(o.choices, 1):
        console.print(f"  [cyan]{i}.[/] {c.label}")


def run_turn(eng: GameEngine, prep):
    if prep.check:
        console.print(f"[bold yellow]{prep.check.badge()}[/]")
    console.print()
    for piece in eng.narrate(prep):
        console.out(piece, end="", highlight=False)
    console.print("\n")
    for line in prep.notes:
        console.print(f"[dim]{line}[/]")
    if eng.encounter and not eng.encounter.over:
        console.print(f"[bold red]⚔ {eng.encounter.status_text()}[/]")
    o = eng.last_outcome
    for i, c in enumerate(o.choices, 1):
        console.print(f"  [cyan]{i}.[/] {c.label}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theme")
    ap.add_argument("--name", default="Vex")
    ap.add_argument("--cls", default="Rogue", choices=list(CLASSES))
    ap.add_argument("--story", type=int, help="resume a story id")
    ap.add_argument("--world", type=int, help="new story inside an existing world id")
    ap.add_argument("--auto", type=int, default=0, help="autoplay N turns")
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO if a.v else logging.WARNING, format="%(name)s: %(message)s")

    eng = GameEngine()
    ok, msg = eng.llm.health()
    console.print(f"[{'green' if ok else 'red'}]{msg}[/]")
    if not ok:
        sys.exit(1)

    if a.story:
        eng.load_story(a.story)
        console.print(f"[bold]Resuming story {a.story}[/]")
        show_outcome(eng, streamed=False)
    else:
        theme = a.theme or Prompt.ask("Theme for your adventure")
        with console.status("") as status:
            eng.new_story(theme, a.name, a.cls, world_id=a.world, progress=lambda m: status.update(m))
        st = eng.store.get_story(eng.story_id)
        console.print(Panel(f"[bold]{st['title']}[/]\n{eng.bible.get('setting', '')}", title="New adventure"))
        run_turn(eng, eng.prepare_opening())

    turns = 0
    while True:
        if a.auto:
            if turns >= a.auto:
                break
            choice = random.choice(eng.last_outcome.choices)
            console.print(f"\n[magenta]> (auto) {choice.label}[/]")
            action, ch = choice.text, choice
        else:
            raw = Prompt.ask("\n[bold]Your move[/] (number, free text, /sheet /clocks /memory /quit)")
            if raw in ("/quit", "/q"):
                break
            if raw == "/sheet":
                console.print(eng.character.sheet_text())
                continue
            if raw == "/clocks":
                from loreweaver.clocks import describe
                for k in eng.clocks.active(eng.world_id):
                    console.print(f"⏱ {describe(k)} [{k.state.get('status')}]")
                continue
            if raw == "/memory":
                eng.wait_ready()
                console.print(Markdown(eng.last_recall.context if eng.last_recall else "(nothing)"))
                console.print(eng.store.stats(eng.world_id))
                continue
            ch = None
            if raw.strip().isdigit() and 1 <= int(raw) <= len(eng.last_outcome.choices):
                ch = eng.last_outcome.choices[int(raw) - 1]
                action = ch.text
            else:
                action = raw
        if eng.busy:
            with console.status("Chronicler is recording memories…"):
                eng.wait_ready()
        with console.status("Consulting the dice and the memory graph…"):
            prep = eng.prepare_turn(action, ch)
        run_turn(eng, prep)
        turns += 1

    with console.status("Finishing memory commit…"):
        eng.wait_ready()
    console.print(f"\nStory id {eng.story_id}, world id {eng.world_id}. Memory: {eng.store.stats(eng.world_id)}")
    console.print(f"LLM: {eng.llm.stats.calls} calls, {eng.llm.stats.seconds:.0f}s · by agent: "
                  + ", ".join(f"{k} {v['calls']}×/{v['seconds']:.0f}s" for k, v in eng.llm.stats.by_agent.items()))
    console.print(f"Structured calls: {eng.llm.stats.json_calls}, constrained-JSON fallbacks: "
                  f"{eng.llm.stats.json_failures}")
    clocks = eng.store.nodes(eng.world_id, types=("clock",))
    console.print("Clocks: " + "; ".join(f"{k.name} {k.state.get('filled')}/{k.state.get('segments')} "
                                         f"{k.state.get('status')}" for k in clocks))
    fixes = [c for c in eng.continuity_log if c["applied"]]
    console.print(f"Continuity: {len(eng.continuity_log)} scenes checked, {len(fixes)} repaired, "
                  f"{sum(len(c['hints']) for c in eng.continuity_log)} rule hints")
    meta = eng.store.get_story(eng.story_id)["meta"]
    console.print(f"Encounters: {meta.get('past_encounters', [])}")


if __name__ == "__main__":
    main()

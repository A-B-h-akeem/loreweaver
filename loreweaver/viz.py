"""Interactive knowledge-graph rendering (pyvis / vis-network).

All labels come from LLM output, so every string is sanitized (see _esc) before it reaches the iframe.
"""
from __future__ import annotations

from typing import Iterable

from pyvis.network import Network

from .memory import MemoryStore

TYPE_STYLE = {
    "player":    ("#f2c14e", "star", "Hero"),
    "character": ("#e07a5f", "dot", "Character"),
    "creature":  ("#c1121f", "triangle", "Creature"),
    "location":  ("#3d9970", "square", "Location"),
    "item":      ("#4ea8de", "diamond", "Item"),
    "faction":   ("#9d4edd", "hexagon", "Faction"),
    "lore":      ("#8d99ae", "dot", "Lore"),
    "event":     ("#adb5bd", "dot", "Event"),
    "thread":    ("#ffb703", "triangleDown", "Plot thread"),
    "decision":  ("#ff6b9a", "diamond", "Decision"),
    "clock":     ("#ff8c42", "triangle", "Clock"),
    "segment":   ("#5c677d", "dot", "Scene"),
    "summary":   ("#7f8fa6", "box", "Summary"),
    "story":     ("#ffd166", "ellipse", "Story"),
}
HIDDEN_RELS_DEFAULT = {"mentions", "next", "from_scene", "made_in"}


def _esc(s: str, n: int = 400) -> str:
    """Labels are drawn on a canvas (entities would show literally) but are embedded as JSON inside a
    <script> tag, so the danger is '</script>' breakouts: strip angle brackets and ampersands."""
    return (s or "")[:n].replace("<", "‹").replace(">", "›").replace("&", "+")


def build_graph_html(store: MemoryStore, world_id: int, *, types: Iterable[str], story_id: int | None = None,
                     highlight: Iterable[int] = (), player_id: int | None = None, dark: bool = True,
                     show_structural: bool = False, height: int = 640) -> tuple[str, int, int]:
    types = set(types)
    nodes = [n for n in store.nodes(world_id) if n.type in types]
    if story_id is not None:
        keep = {r for r in (n.id for n in nodes) if story_id in store.node_story_ids(r)}
        nodes = [n for n in nodes if n.id in keep or n.story_id == story_id]
    ids = {n.id for n in nodes}
    hl = set(highlight)
    fg = "#e6e6e6" if dark else "#222222"
    net = Network(height=f"{height}px", width="100%", bgcolor="rgba(0,0,0,0)", font_color=fg,
                  directed=True, cdn_resources="in_line")
    net.barnes_hut(gravity=-9000, central_gravity=0.25, spring_length=140, spring_strength=0.02, damping=0.35)

    for n in nodes:
        color, shape, label = TYPE_STYLE.get(n.type, ("#999999", "dot", n.type))
        name = n.name if n.type not in ("event", "decision", "segment") else (n.name[:38] + "…" if len(n.name) > 38 else n.name)
        tip_lines = [f"{label}: {n.name}"]
        if n.status:
            tip_lines.append(f"status: {n.status}")
        if n.description:
            tip_lines.append(n.description[:300])
        if n.aliases:
            tip_lines.append("aka " + ", ".join(n.aliases[:4]))
        tip_lines.append(f"introduced turn {n.turn}, importance {n.importance:.1f}")
        is_gone = n.status.lower() in ("dead", "destroyed", "slain", "killed")
        net.add_node(n.id, label=_esc(name, 60), title=_esc("\n".join(tip_lines), 700), shape=shape,
                     color={"background": "#444444" if is_gone else color,
                            "border": "#ffffff" if n.id in hl else color,
                            "highlight": {"background": color, "border": "#ffffff"}},
                     borderWidth=4 if n.id in hl else 1, size=8 + 2.2 * n.importance,
                     font={"color": fg, "size": 13 if n.type in ("event", "segment", "decision") else 16})

    n_edges = 0
    for e in store.edges(world_id):
        if e.src not in ids or e.dst not in ids:
            continue
        if not show_structural and e.rel in HIDDEN_RELS_DEFAULT:
            continue
        col = "#d1495b" if e.sentiment < 0 else "#66c2a5" if e.sentiment > 0 else "#7d8597"
        label = e.rel.replace("_", " ")
        title = label + (f" ({e.sentiment:+d})" if e.sentiment else "") + f" · turn {e.updated_turn}"
        net.add_edge(e.src, e.dst, label=_esc(label, 30), title=_esc(title, 80), color=col,
                     width=1 + min(4.0, abs(e.sentiment) / 3 + e.weight / 3), arrows="to",
                     font={"color": fg, "size": 10, "strokeWidth": 0})
        n_edges += 1
    if not nodes:
        return "", 0, 0
    return net.generate_html(notebook=False), len(nodes), n_edges

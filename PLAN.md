# Loreweaver — AI Dungeon Master with Graph Memory

**Problem statement (M1-P8):** an autonomous storytelling system. The user gives a theme and the AI
generates a continuously evolving story. It remembers previously introduced characters and events,
and later story segments change based on the user's decisions.

**Our take:** a single-player, D&D-style tabletop game run by a local LLM acting as Dungeon Master.
It is backed by a **temporal knowledge-graph memory with hybrid RAG (GraphRAG-lite)**. Every story,
character, event, decision and consequence is a node in one connected graph, and stories set in the
same world share that graph.

---

## 1. Hardware & model decisions (benchmarked on this machine)

| | Choice | Why |
|---|---|---|
| GPU | RTX 5060 Laptop, 8 GB VRAM, 16 GB RAM | |
| LLM | `qwen3.5:4b` (Q4_K_M, 262K native ctx, hybrid linear attention) | ~67 tok/s; best extraction quality in our tests; small KV cache → long context is cheap |
| Context | `num_ctx = 16384` | 6.4 GB VRAM incl. KV cache |
| Embeddings | `qwen3-embedding:0.6b` (1024-d), `num_ctx = 2048` | 1.2 GB; both models stay resident (7.2 GB total) → no swapping between turns |
| Thinking | **off** (`think=False`) | thinking took 36–44 s per JSON call; off takes ~8 s |
| Structured output | JSON schema in the prompt → pydantic validate → `json-repair` → 1 retry with the error fed back | Ollama's `format=` grammar is not enforced for qwen3.5 when `think=False` (verified) |

Rejected: `qwen3:1.7b` (fast but garbles names: "LIGTHHOUSE", ALL-CAPS), `qwen2.5:7b` (6.3 GB, which
evicts the embedder; sparse extraction).

---

## 2. Architecture

```
            ┌──────────────────────── Streamlit UI ─────────────────────────┐
            │ Story · Choices · Dice · Sheet · Graph · Codex · Timeline ·   │
            │ Recall Inspector · Director                                    │
            └───────────────────────────────┬────────────────────────────────┘
                                            │
                                   GameEngine (orchestrator)
     ┌──────────────┬──────────────┬────────┴──────┬──────────────┬──────────────┐
 Worldbuilder   Adjudicator     Narrator (DM)   Chronicler    Director       Summarizer
 theme→world    action→check    streams scene   scene→facts   arc plan,      hierarchical
 bible + NPCs   (ability, DC)   + 3 choices     (JSON)        callbacks,     chapter/arc
                    │                ▲               │        consequences   summaries
                 Rules (d20)         │               ▼
                                MemoryRetriever ◄─ MemoryStore (SQLite + FTS5 + vectors)
                                (hybrid + graph)        │
                                                  StoryGraph (networkx, PPR)
```

### Agents (all one local model, different prompts)
1. **Worldbuilder:** turns the theme into a world bible: setting, tone, central conflict, 3–4
   locations, 2 factions, 3–4 NPCs (with secrets and goals), opening hook. If the story continues an
   *existing world*, it receives that world's notable entities and history and ties into them.
2. **Adjudicator:** for free-text actions, decides whether a check is needed, which ability and what
   DC. Choices the narrator offers come pre-tagged (`[DEX 14]`), so they skip this call.
3. **Rules engine (pure Python, no LLM):** D&D-lite. Six abilities, classes, d20 + modifier +
   proficiency vs DC, natural 20/1 crits, outcome tiers (strong success / success / success-at-a-cost
   / failure), damage rolls, HP, XP and levels, inventory, gold. Dice are rolled by code so they are fair.
4. **Narrator (Dungeon Master):** writes the next scene (streamed) from a structured context
   (§3.4) and ends at a decision point with 3 choices.
5. **Chronicler:** reads the action and the scene and extracts structured memory: new or updated
   entities, state changes (e.g. *dead*, *captured*, *location*), relations with sentiment, events
   with importance, plot threads opened/advanced/resolved, **player decisions with their likely
   future consequences**, player HP/items/gold/XP changes. Runs **in a background thread** while the
   player reads.
6. **Director:** every few turns it re-plans the arc: act, tension curve, the next 3 beats,
   **callbacks** (important but dormant characters to bring back) and **consequences that are due**.
   Its notes steer the narrator, so the story moves forward on its own instead of only reacting.
7. **Summarizer:** every 5 turns it writes a chapter summary. When there are more than 8 chapters, it
   rolls them into an arc summary. "Story so far" stays a fixed size no matter how long the game gets.

---

## 3. Memory system (the core)

### 3.1 One graph for everything — `nodes` + `edges` in SQLite
Node types: `player, character, creature, location, item, faction, lore, event, thread, decision,
segment, summary, story`.

Edge examples:
- `character —ally_of/enemy_of/loves/fears (sentiment −10..10)→ character`
- `character —located_at→ location`, `player —owns→ item`, `character —member_of→ faction`
- `event —involves→ entity`, `event —occurred_at→ location`, `event —caused→ event`
- `decision —affects→ entity`, `decision —led_to→ event` (cause and consequence chains)
- `thread —concerns→ entity`, `event —advances/resolves→ thread`
- `segment —mentions→ entity`, `segment —next→ segment`, `summary —summarizes→ segment`
- `entity —appears_in→ story`. Worlds hold many stories, so **stories are interconnected**: a
  villain from campaign 1 can return in campaign 2 with their scars, grudges and history intact.

### 3.2 Temporal state
Every node has a `state` (status, location, mood, …) and a **history table** of changes by turn,
for example "Silas: status alive→dead @ turn 12". The narrator gets current state plus canon
constraints ("dead characters stay dead unless the story explains it").

### 3.3 Retrieval pipeline (hybrid + graph, per turn)
1. **Query** = player action + tail of the last scene + current location + active thread names.
2. **Exact recall:** known names and aliases mentioned in the query become strong seeds.
3. **Dense:** cosine similarity over embeddings of entity cards, events, segments, summaries and lore
   (numpy matrix cached in memory).
4. **Sparse:** SQLite FTS5 BM25 for exact names and rare words that embeddings blur.
5. **Reciprocal Rank Fusion** of dense and sparse results. A **relation channel** matches the query
   against edge labels and sentiment ("who distrusts me?"); it was added after evaluation, see README.
6. **Graph expansion:** Personalized PageRank on the story graph, seeded by steps 2–5 plus the player
   and the current location, normalized by sqrt(degree) against hub bias. Pulls in *connected* memories the query never mentioned (the
   betrayed ally, the faction behind the attack, the item's original owner).
7. **Scoring:** `0.40·relevance + 0.30·graph + 0.15·importance + 0.15·recency`, plus boosts for
   exact mentions, open threads and pending consequences.
8. **Packing within a token budget** into sections: *Characters & entities present* (cards with
   state and key relations), *Relevant past events* (chronological, with turn numbers), *Recalled
   passages*, *Open threads*, *Pending consequences*, *Echoes from other stories in this world*.

Every retrieval is logged and shown in the **Recall Inspector**: what was recalled, why, and with
what score.

### 3.4 Narrator context layout (≈16K budget)
System rules (~700) → World bible (~500) → Hero sheet (~200) → Story-so-far summaries (~1200) →
Memory recall (~3000) → Director notes (~300) → Last 3 scenes verbatim (~2500) → Action + dice result
→ output (~700).

### 3.5 Entity resolution (keeps the graph clean)
exact name/alias → normalized token-subset match ("Harbormaster Silas" = "Silas") → fuzzy string
(≥0.88) → embedding similarity (≥0.92, same type) → otherwise create. Matched surface forms become
aliases. The chronicler is also given canonical names of the entities in the scene so it reuses them.

### 3.6 Decisions → consequences
Each decision is stored as a `decision` node with `affects` edges and a pending consequence. NPC
attitude toward the player accumulates as edge sentiment. The Director schedules pending
consequences, retrieval boosts them, and the Chronicler marks them `resolved` when they pay off.
This is how earlier choices change later segments.

---

## 4. Turn lifecycle
1. Wait for the previous background commit (usually already done).
2. Action: choice (pre-tagged check) or free text (Adjudicator).
3. Rules roll the dice and work out the outcome and damage.
4. Retrieve memory (§3.3) and add Director notes.
5. Narrator streams the scene and choices; save the segment and turn immediately.
6. Background: Chronicler → entity resolution → graph upserts, events, decisions, threads, player
   state → embed changed nodes → (every 5 turns) Summarizer → (every 4 turns or on thread
   resolution) Director re-plan.

---

## 5. Project layout
```
loreweaver/
  config.py        settings (pydantic-settings, env-overridable)
  llm.py           Ollama wrapper: stream, chat, robust JSON (validate/repair/retry), embeddings
  schemas.py       pydantic models (WorldBible, Extraction, ArcPlan, Adjudication, …)
  prompts.py       all agent prompts
  rules.py         dice, character sheet, checks
  memory/
    store.py       SQLite schema, CRUD, FTS5, vectors, history
    graph.py       networkx view, Personalized PageRank
    resolver.py    entity resolution
    retriever.py   hybrid retrieval, scoring, context packing
  agents/
    worldbuilder.py adjudicator.py narrator.py chronicler.py director.py summarizer.py
  engine.py        GameEngine orchestration and background commit
app.py             Streamlit UI
cli.py             terminal client (debugging / low-resource)
tests/             unit tests with fake LLM and embedder; engine pipeline test
eval/recall_eval.py automated playthrough + memory-recall benchmark
```

## 6. Build order
1. config, schemas, llm wrapper (robust JSON)
2. rules engine + tests
3. memory store, graph, resolver, retriever + tests (fake embedder)
4. agents + prompts
5. engine + fake-LLM pipeline test
6. CLI, then a smoke test against real Ollama
7. Streamlit UI (story, sheet, graph, codex, timeline, inspector, director)
8. Recall evaluation script, README

Status: phase 1 is complete (18 tests, recall Hit@5 90% on a held-out story). Phase 2 is complete (41 tests; see README for evaluation). Phase 3 is complete (44 tests; results in §13 and the README). Final real-model smoke run on 2026-11-07: 6 autoplay turns, 0 JSON fallbacks, the UI renders without errors.

---

# Phase 2 — fixing the limitations

## 7. What research says about each limitation

| Limitation seen in phase 1 | Research / prior art | Takeaway |
|---|---|---|
| The 4B narrator bends established facts | NCP-Bench (arXiv 2608.08160): fact conflicts end 40–68% of interactive-narrative runs, and even the best model stays conflict-free in only 42% of runs after 20 turns. The cause is the lack of explicit state tracking. ConStory-Bench (2603.05890): factual/detail and timeline errors dominate, and its checker works by extracting spans and pairing them with **quoted evidence**. "Seduced by the narrative" (2607.02802): models bend rules to serve the story, and explicit state injection plus external rule engines help most | Give the narrator an **authoritative ledger** of current state at the end of the prompt. **Ground every extracted fact** in the scene text. Run a **checker that must quote evidence**, then repair. Most importantly, never let a contradiction **enter memory**, because errors in memory propagate |
| Predicted consequences are vague and "resolve" too early | Blades in the Dark **progress clocks** (an effect level fills a set number of segments; when the clock is full, the thing happens). Dungeon World **fronts / grim portents** (escalating threat steps). Drama-manager literature: delayed consequence needs robust state tracking | Turn each consequence into a **clock** with a concrete payoff. The LLM proposes clocks and ticks them, but **code** decides when one fires, and then the payoff is mandatory in the next scene. The LLM no longer judges resolution |
| Combat is one narrative check | Setting the DC (NeurIPS 2025): a tool-grounded 5e simulator in which the LLM acts through function calls and the engine enforces rules and state. "A Dungeon Master as a long-horizon agent" (h-tu.ch): game logic lives in deterministic tools, and state is split into **phases** (combat state is archived when combat ends) | An **encounter engine** in code: enemy stat blocks by tier, attack and defence rolls, enemy turns, morale, healing, fleeing. The narrator only describes results the engine has already computed, and the engine writes the outcome (who died or fled) to memory |
| JSON needs repair and retries | Ollama issue #14645: format masking waits for `</think>`, so it never engages with `think=False` on `/api/chat` (v0.17.6–0.30.7, fixed in 0.31.2). **Verified locally on 0.23.2:** `/api/generate` does enforce the schema grammar | Structured agents call `/api/generate` with `format=<schema>`, keeping validate/repair as a fallback |

Sources: arxiv.org/html/2608.08160 · arxiv.org/html/2603.05890 · arxiv.org/pdf/2607.02802 ·
h-tu.ch/blog/dungeon-master-long-horizon-agent · neurips.cc/virtual/2025/loc/san-diego/128312 ·
slyflourish.com/progress_clocks_in_dnd.html · a-dungeon-world.fandom.com/wiki/Fronts · github.com/ollama/ollama/issues/14645

## 8. Phase 2 design

### A. Grammar-constrained JSON
`OllamaLLM.json()` calls `generate(format=schema, think=False)`, then pydantic validation, then the repair-and-retry fallback. Track the invalid-JSON rate in stats.

### B. Consistency: ground → guard → check → repair
1. **Scene ledger** (pre-generation), placed last in the narrator prompt as "LEDGER — authoritative, never contradict":
   - characters present at the hero's location, with status and attitude toward the hero
   - characters known to be elsewhere, dead or missing
   - hero HP and inventory ("the hero has ONLY these items")
   - the active encounter state
   - clocks that are close to firing or due now
2. **Evidence-grounded extraction:** an extracted entity, relation participant, event participant or item is kept only if its name (or an alias, or a distinctive token) appears in the action or narration. This stops the Chronicler from hallucinating into memory.
3. **Canon guards at commit:**
   - Dead or destroyed → alive is refused unless the text justifies it (resurrected, revived, survived, faked…).
   - An item the hero lost or never had cannot be "used".
   - The player's HP is owned by the rules engine during combat.
4. **Continuity checker** (background, after the commit):
   - Rule checks against the ledger: dead characters acting, phantom items, absent characters speaking.
   - An LLM check that returns `{violations: [{fact, quote, fix}]}`. A violation is accepted only if its quote really occurs in the scene.
5. **Auto-repair:** the LLM rewrites only the offending sentences with the correct fact. The stored scene and segment are replaced (and re-embedded), and the UI shows "continuity fix applied".

### C. Consequence clocks (fronts)
- **New node type `clock`:** name, kind (`danger` or `progress`), segments (4, 6 or 8), filled count, payoff (a concrete event), status (`ticking`, `due`, `fired`, `averted`). Edges: `decision —started→ clock`, `clock —threatens→ entity`.
- **Sources:**
  - Each hero decision gets a danger clock (severity 1–2 → 4 segments, 3–4 → 6, 5 → 8).
  - The worldbuilder creates a **front**: the central threat as an 8-segment clock with grim portents.
- **Ticks:**
  - The Chronicler reports `clock_ticks` (name, 1–2, reason) when the scene pushes a threat forward.
  - The rules tick the most relevant danger clock on a failure (+1, or +2 on a critical failure).
  - The Director ticks the front by +1 at each re-plan, so time passes.
  - The Chronicler can mark a clock **averted** when the hero clearly neutralizes it, but only with a quote.
- **Firing:** a full clock becomes `due`. The next narrator prompt gets "THIS SCENE MUST: <payoff>". After that scene the clock is `fired`, the decision becomes `resolved`, and `led_to` edges point to that scene's events. Resolution is deterministic, so it can't happen too early.

### D. Encounter engine (combat phase)
- **Start:** the Chronicler reports `combat_starts: [{name, tier}]` when hostiles attack or the hero attacks. The engine creates an encounter with stat blocks:

  | tier | HP | DEF | attack | damage |
  |---|---|---|---|---|
  | minion | 4 | 10 | +2 | d4 |
  | standard | 10 | 12 | +3 | d6 |
  | elite | 20 | 14 | +5 | d8 |
  | boss | 40 | 15 | +6 | d10, two attacks |

  Stats scale with the hero's level.
- **Hero AC:** 10 + DEX modifier + class armour (Fighter +6, Cleric +4, Ranger/Rogue/Bard +2, Wizard +0).
- **Choices come from code:**
  - Attack <enemy> (the class's best ability)
  - Defend (+4 AC, and the next attack has advantage)
  - Use a healing potion (2d4+2, if one is in the inventory)
  - Flee (DEX vs a DC that rises with the number of enemies)
  - Parley (CHA, only once someone is bloodied)
  - Free text is classified into one of these.
- **A round:**
  1. The hero acts (d20 + bonus vs DEF; damage is the class die + modifier; a crit doubles the dice).
  2. Enemies make morale checks when bloodied or when their leader falls, and may flee or surrender.
  3. Each enemy still standing attacks (d20 + attack vs AC).
  4. The narrator describes the computed results.
- **End:** when all enemies are defeated, fled or surrendered, or the hero flees or is downed. **Code** writes enemy statuses (dead, fled, captured) and a summary event to memory, awards XP by tier, and archives the encounter (phase compaction).

### E. Evaluation
- `eval/consistency_eval.py`, adversarial probes in the style of NCP-Bench, run on a copy of a played story:
  - ask a dead character for help
  - use an item the hero doesn't have
  - greet a hostile NPC as an old friend
  - claim a false past event ("as Silas promised me yesterday…")
- Each probe is judged by rule checks plus the evidence-quoting checker, reporting the violation rate with each mitigation on and off.
- Re-run `recall_eval.py` (no regressions). Report clock statistics: how many clocks fired and how many turns passed from decision to payoff.

## 9. Phase 2 build order
1. A: grammar JSON (+ test)
2. B2/B3: grounding and guards in the Chronicler (+ tests)
3. C: clocks: store, Chronicler, rules, Director, engine firing (+ tests)
4. D: encounter engine in rules/engine, combat choices (+ tests)
5. B1 + B4/B5: ledger, checker and repair (+ tests)
6. UI: clocks, encounter panel, continuity notices. CLI parity.
7. Real-model smoke runs, consistency eval with ablation, recall eval, README update

## 10. Lessons from real-model runs (phase 2 iteration)

Two 12-turn autoplay runs on `qwen3.5:4b` changed the design:

| Observed | Cause | Change |
|---|---|---|
| The campaign front fired by turn 6 | It was the only clock, so it was ticked almost every scene | Scene-tick cooldown (2 turns, 4 for fronts); scenes nudge a front by at most 1; only critical failures feed a front; a front cannot fire before turn 18 (`front_min_turn`) |
| 4 decisions, only 1 clock | Decision clocks were named after the front and deduplicated away | A decision clock never merges into a front; repeated threats tick the existing clock and get linked to it |
| 8 of 12 scenes "repaired" | The checker treated in-scene developments as contradictions, and stale locations triggered "elsewhere" flags | Violations carry a `kind`; only `dead_acting` and `missing_item` are auto-repaired, each verified deterministically (name + action verb; second person and not a carried item) |
| Fights never used the engine | Hero-initiated fights were narrated in prose | Combat intent on the action's **leading verb** → opponents call → encounter starts before narration |
| A boss fight at level 1; a second fight right after a knockout | Tiers came from the LLM; the intent regex matched "…finishes its strike" | Difficulty budget (cost ≤ 4 + 2·(level−1); bosses only in act 3 or at level 4+); 3-turn cooldown after a knockout; leading-verb regex |
| The opening shattered the hero's shield and took them to 2 HP | Dramatic opening plus extraction of narrative damage | The opening can never cost HP or gear; narrative-only damage outside combat is capped at −3 |
| Constrained JSON: **0 fallbacks** in 61 structured calls | `/api/generate` grammar works | Kept |
| Adversarial eval: post-hoc repair alone left 50% contradictions; phantom items at 100% | The narrator follows the false premise in the *player's action* whatever the ledger says, and sentence fixes can't save a scene built on it; the verb list missed "moves"/"gasps" | **Action precheck** (rules-level correction before narration), a verb-agnostic dead-acting detector, and a scene-rewrite fallback → 6–12% overall, phantom items 0–25% |

---

# Phase 3 — efficiency and memory quality

## 11. Measurements that drive phase 3
Per-call profile on a real turn (qwen3.5:4b, RTX 5060):
- **Narrator:** ~5,700 prompt tokens take **4–5 s before the first word**. Ollama reuses a cached prefix
  (3.77 s → 1.59 s) only if no other call intervenes, and our turn always interleaves checker, Chronicler and adjudicator.
- **Chronicler:** ~880 output tokens take **13 s**, because grammar-constrained output is pretty-printed JSON that
  restates empty and unchanged fields. An A/B test on 5 real scenes: compact JSON gives **−38% tokens and −37% time**
  with the same validity and the same or richer extraction. GraphRAG-style tuples give −51% but lost relations in one scene.

## 12. Research → design
| Idea | Source | Design |
|---|---|---|
| Compact structured output | A/B above; GraphRAG uses delimiter tuples to save tokens | Every structured call asks for minified one-line JSON and omits empty or unchanged fields; the grammar still guarantees structure |
| Prefix KV-cache reuse | llama.cpp/Ollama per-slot prefix caching (measured locally) | Reorder the narrator prompt so its stable part comes first (system, world, story so far, recent scenes); after each commit, **warm the cache** with that prefix while the player reads |
| Risk-gated verification | ConStory-Bench: trigger verification on risk signals rather than always | Run the continuity checker only when the scene names a dead character, the precheck fired, or the hero uses something |
| Query-to-triple retrieval | HippoRAG 2 (arXiv 2502.14802): matching the query to fact triples beats entity matching for associative recall | A **fact channel**: embed relation edges as sentences ("Silas Crane hates Kael") and seed both endpoints of the best-matching facts |
| Query-aware spreading activation | arXiv 2606.30133: gating propagation by each node's query similarity, σ(v)=max(cos,0), +3–7 F1 over PPR | A **gated PageRank**: edge weights into a node are scaled by that node's query gate (with a floor), so relevance flows toward query-relevant memories |
| Temporal edge invalidation | Zep/Graphiti bi-temporal KG (arXiv 2501.13956) | When a new relation between the same pair contradicts an old one (sentiment sign flip), the old edge gets `valid_to`; cards and graph use current facts and the history stays queryable |

Evaluation: recall_eval ablation on three played stories (an 8-turn dev story, the 15-turn held-out story, the 12-turn
phase-2 story); latency measured on real turns before and after.

## 13. Phase 3 results
**Retrieval** (recall_eval, 3 played stories, 195 probes; Hit@5 / Hit@10 / MRR / packed):

| config | Hit@5 | Hit@10 | MRR | packed |
|---|---|---|---|---|
| phase-2 (named+dense+BM25+relation, PPR) | 81.5% | 96.4% | 0.580 | 99.5% |
| **gated PPR (query-aware), new default** | **83.6%** | 95.9% | **0.590** | 99.5% |
| spreading activation (T=3, α=0.7) | 84.6% | 94.9% | 0.559 | 99.5% |
| + fact seeds (HippoRAG-2 triples as walk seeds) + gated PPR | 84.6% | 95.9% | 0.586 | 99.5% |
| + facts as boosts (negative result) | 72.8% | 92.3% | 0.525 | 99.5% |
| classic hybrid (dense+BM25) | 73.8% | 86.2% | 0.475 | 88.2% |

- **Gated PPR** improved Hit@5 and MRR on all three stories at negligible cost, so it is the new default.
- **Spreading activation** ranks the target into the top 5 more often, but lower within it.
- **Fact boosts flood the ranking** with generic relation partners. Facts used as walk seeds are neutral within noise, so they remain opt-in.

**Efficiency:**
- **Chronicler A/B:** compact JSON gives −38% tokens and −37% decode time.
- **Latency:** time to first token fell from **4.5 s to 2.3 s** on choice turns (prefix warm-up).
- **Checker gate:** the tightened risk signal fires on 0 of 27 ordinary scenes, so the checker runs only when a dead character is named, the precheck fired, or the hero uses an item.

**Memory:** temporal invalidation of contradicted relations (`valid_to`). Entity cards and the graph walk prefer current facts, and the Codex shows when a relation ended.

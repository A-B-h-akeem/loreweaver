"""Runtime settings. Every field can be overridden with an env var prefixed LOREWEAVER_."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LOREWEAVER_", env_file=".env", extra="ignore")

    # --- models (benchmarked on an 8 GB RTX 5060, see PLAN.md) ---
    ollama_host: str = "http://localhost:11434"
    llm_model: str = "qwen3.5:4b"
    embed_model: str = "qwen3-embedding:0.6b"
    num_ctx: int = 16384
    embed_ctx: int = 2048
    keep_alive: str = "30m"

    # --- sampling ---
    narrator_temperature: float = 0.85
    creative_temperature: float = 0.9
    extract_temperature: float = 0.1
    narrator_max_tokens: int = 900

    # --- storage ---
    db_path: Path = ROOT / "data" / "loreweaver.db"

    # --- memory / retrieval ---
    recent_scenes: int = 3                # verbatim scenes in narrator context
    retrieve_candidates: int = 24          # per channel (dense / sparse)
    memory_token_budget: int = 2000        # 2000 packs 100% of probe targets vs 99.5% at 3000 (PLAN.md §13)
    summary_token_budget: int = 1200
    rrf_k: int = 60
    retrieval_channels: tuple = ("named", "dense", "bm25", "relation", "graph")   # chosen by recall_eval
    graph_mode: str = "gated_ppr"          # ppr | gated_ppr | spread — gated won on 3 stories (PLAN.md §13)
    resolve_fuzzy: float = 0.88
    resolve_embed: float = 0.92

    # --- consistency (PLAN.md §8B; switchable for ablation in eval/consistency_eval.py) ---
    use_ledger: bool = True                # authoritative state block at the end of the narrator prompt
    action_precheck: bool = True           # rules-level correction of false premises in the player's action
    continuity_check: bool = True          # evidence-quoting checker + sentence-level repair

    # --- efficiency (PLAN.md §11-12) ---
    compact_json: bool = True              # minified one-line structured output (-38% decode tokens)
    warm_prefix: bool = True               # pre-compute the next narrator prompt's stable prefix after each commit
    gate_checker: bool = True              # run the continuity checker only on risk signals

    # --- consequence clocks ---
    front_min_turn: int = 18               # a campaign front cannot fire before this turn

    # --- cadence ---
    summarize_every: int = 5               # turns per chapter summary
    arc_rollup_after: int = 8              # chapters before rolling into an arc summary
    replan_every: int = 4                  # director re-plan cadence


settings = Settings()

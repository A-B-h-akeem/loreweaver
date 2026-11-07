import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loreweaver.memory import EntityResolver, MemoryRetriever, MemoryStore, StoryGraph  # noqa: E402


def fake_embed(texts: list[str], dim: int = 256) -> np.ndarray:
    """Deterministic bag-of-words hashing embedder: similar words -> similar vectors."""
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in re.findall(r"[a-z]{3,}", t.lower()):
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            out[i, h % dim] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norms, 1e-9)


@pytest.fixture
def store():
    s = MemoryStore(":memory:")
    yield s
    s.close()


@pytest.fixture
def mem(store):
    graph = StoryGraph(store)
    return store, graph, EntityResolver(store, fake_embed), MemoryRetriever(store, graph, fake_embed)


def embed_all(store, world_id):
    stale = store.stale_embeddings(world_id)
    vecs = fake_embed([n.text() for n in stale])
    store.set_embeddings([(n.id, vecs[i], store.text_hash(n.text())) for i, n in enumerate(stale)])

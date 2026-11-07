from types import SimpleNamespace

from loreweaver.llm import OllamaLLM
from loreweaver.schemas import Adjudication


class FakeClient:
    def __init__(self, generate_text: str, chat_text: str):
        self.generate_text, self.chat_text = generate_text, chat_text
        self.calls = []

    def generate(self, **kw):
        self.calls.append(("generate", kw))
        return SimpleNamespace(response=self.generate_text, eval_count=10)

    def chat(self, **kw):
        self.calls.append(("chat", kw))
        return SimpleNamespace(message=SimpleNamespace(content=self.chat_text), eval_count=10)


def _llm(client) -> OllamaLLM:
    llm = OllamaLLM()
    llm.client = client
    return llm


def test_json_uses_grammar_constrained_generate_first():
    client = FakeClient('{"needs_check": true, "ability": "DEX", "dc": 14}', "unused")
    out = _llm(client).json("sys", "prompt", Adjudication)
    assert out.needs_check and out.dc == 14
    kind, kw = client.calls[0]
    assert kind == "generate" and kw["format"]["properties"]["ability"] and kw["think"] is False
    assert len(client.calls) == 1


def test_json_falls_back_to_free_generation_and_repair():
    client = FakeClient('{"needs_check": tr', 'Sure: {"needs_check": false, "dc": 9}')
    llm = _llm(client)
    out = llm.json("sys", "prompt", Adjudication)
    assert out.needs_check is False and out.dc == 9
    assert [c[0] for c in client.calls] == ["generate", "chat"]
    assert llm.stats.json_failures == 1

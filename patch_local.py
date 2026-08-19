"""Compatibility shim: agentdojo 0.1.30 <-> modern vLLM OpenAI server.

agentdojo sends chat content parts as {'type':'text','content':...} (old shape).
vLLM 0.27's OpenAI server validates against the current schema, which requires
{'type':'text','text':...}, so every request 400s, the model never runs, and the
benchmark reports a spurious 0% ASR.

LocalLLM uses a text protocol (no native tool-calls), so flattening list-content to
a plain string is lossless and accepted. Patch the single chokepoint that sends to
the server. Load INSIDE the benchmark subprocess with:

    PYTHONPATH="$PWD" python -m agentdojo.scripts.benchmark ... -ml patch_local

(An in-kernel monkeypatch does NOT work — the benchmark runs as a separate process.)
"""
import agentdojo.agent_pipeline.llms.local_llm as _L

_orig = _L.chat_completion_request


def _patched(client, model, messages, **kw):
    fixed = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            c = "".join(
                (p.get("text", p.get("content", "")) if isinstance(p, dict) else str(p))
                for p in c
            )
            m = {**m, "content": c}
        fixed.append(m)
    return _orig(client, model=model, messages=fixed, **kw)


_L.chat_completion_request = _patched
print("[patch_local] content-parts -> string applied inside benchmark process")

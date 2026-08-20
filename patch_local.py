"""Compatibility + correctness shims for agentdojo 0.1.30 <-> modern vLLM + Qwen.

Load INSIDE the benchmark subprocess with:
    PYTHONPATH="$PWD" python -m agentdojo.scripts.benchmark ... -ml patch_local

Fixes (all discovered empirically running Qwen2.5-7B on the banking suite):

1. Content-part schema: agentdojo sends {"type":"text","content":...}; modern vLLM's
   OpenAI server requires {"type":"text","text":...}. Without this every call 400s and
   ASR reads a fake 0%. LocalLLM uses a text tool-protocol, so flattening list content
   to a plain string is lossless. Bound idempotently (the real fn via default arg) so a
   double import/apply can't create an infinite _ccr->_ccr recursion.

2. int<->str digit cap: Python 3.11+ caps conversion at 4300 digits; some banking tool
   outputs carry longer numeric strings and agentdojo's json decode raises ValueError.

3. Tool-call parser: Qwen closes tool calls with a BARE `<function>` (not `</function>`),
   so agentdojo's _parse_model_output grabs `{...}<function>` and JSON-fails -> the agent
   can't act -> utility AND ASR are suppressed (false negatives). We brace-match the JSON
   object and ignore trailing tags. This de-contaminates utility and adaptive ASR.
"""
import sys, re, json
import agentdojo.agent_pipeline.llms.local_llm as _L
from agentdojo.agent_pipeline.llms.local_llm import (
    ChatAssistantMessage, FunctionCall, text_content_block_from_string,
)

sys.set_int_max_str_digits(1_000_000)

# --- 1. content-part shim (idempotent) ---
_REAL_CCR = _L.chat_completion_request
def _ccr(client, model, messages, __real=_REAL_CCR, **kw):
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
    return __real(client, model=model, messages=fixed, **kw)
_L.chat_completion_request = _ccr

# --- 3. robust tool-call parser (brace-match; tolerate Qwen's bare `<function>` close) ---
_OPEN_RE = re.compile(r"<function\s*=\s*([^>]+?)\s*>")

def _extract_json_object(s, start):
    i = s.find("{", start)
    if i == -1:
        return None
    depth = 0; in_str = False; esc = False
    for j in range(i, len(s)):
        c = s[j]
        if in_str:
            if esc: esc = False
            elif c == "\\": esc = True
            elif c == '"': in_str = False
        else:
            if c == '"': in_str = True
            elif c == "{": depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return s[i:j + 1]
    return None

def _robust_parse(completion):
    default = ChatAssistantMessage(
        role="assistant", content=[text_content_block_from_string(completion.strip())], tool_calls=[]
    )
    calls = []
    for m in _OPEN_RE.finditer(completion):
        name = m.group(1).strip()
        obj = _extract_json_object(completion, m.end())
        args = {}
        if obj is not None:
            try:
                parsed = json.loads(obj)
                if isinstance(parsed, dict):
                    args = parsed
            except Exception:
                args = {}
        calls.append(FunctionCall(function=name, args=args))
    if not calls:
        return default
    return ChatAssistantMessage(
        role="assistant", content=[text_content_block_from_string(completion.strip())], tool_calls=calls
    )
_L._parse_model_output = _robust_parse

print("[patch_local] content-part shim + int-digit cap + robust tool-parser applied")

if __name__ == "__main__":
    # self-check on the two observed Qwen failure modes
    def _args(fc): return fc.args if hasattr(fc, "args") else fc["args"]
    a = _robust_parse('<function=read_file>{"file_path": "x.txt"}<function>')
    assert a["tool_calls"] and _args(a["tool_calls"][0]) == {"file_path": "x.txt"}, a
    b = _robust_parse('<function=get_balance></function>')
    assert b["tool_calls"] and _args(b["tool_calls"][0]) == {}, b
    c = _robust_parse('no tool call here')
    assert c["tool_calls"] == []
    print("self-check passed")

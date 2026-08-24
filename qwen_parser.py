"""Canonical, dependency-free parser for Qwen2.5's tool-call text format.

Qwen closes a tool call with a BARE `<function>` tag, not `</function>`. AgentDojo's
built-in parser searches for `</function>`, never finds it, grabs `{...}<function>`, and the
JSON decode fails — so the agent takes no action and both utility and ASR read false-low.
This module brace-matches the JSON object and tolerates the bare close tag.

This is the tested reference implementation of the algorithm used by the in-process shim
(`run_sweep.apply_shims`) and the subprocess shim (`patch_local.py`). Keep the three in sync;
`tests/test_qwen_parser.py` guards this one.
"""
import re
import json

_OPEN_RE = re.compile(r"<function\s*=\s*([^>]+?)\s*>")


def _extract_json_object(s, start):
    """Return the first balanced {...} object at/after `start`, or None. Ignores braces in strings."""
    i = s.find("{", start)
    if i == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for j in range(i, len(s)):
        c = s[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return s[i:j + 1]
    return None  # unbalanced


def extract_tool_calls(completion):
    """Parse a raw completion into a list of {"name": str, "args": dict}.

    Every `<function=NAME>` opener yields one call; its args are the first balanced JSON object
    after the tag (empty dict if none / unparseable / not a dict). Trailing junk like a bare
    `<function>` close is ignored. Returns [] when there is no tool call.
    """
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
        calls.append({"name": name, "args": args})
    return calls

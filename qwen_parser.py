"""Canonical, dependency-free parser for Qwen2.5's tool-call text format.

Qwen closes a tool call with a BARE `<function>` tag, not `</function>`. AgentDojo's
built-in parser searches for `</function>`, never finds it, grabs `{...}<function>`, and the
JSON decode fails — so the agent takes no action and both utility and ASR read false-low.
This module brace-matches the JSON object and tolerates the bare close tag.

Single source of truth: both the in-process shim (`run_sweep.apply_shims`) and the subprocess
shim (`patch_local.py`) import `extract_tool_calls` from here, so there is no hand-maintained
copy to drift. `tests/test_qwen_parser.py` guards it.
"""
import re
import json

_OPEN_RE = re.compile(r"<function\s*=\s*([^>]+?)\s*>")


def _extract_json_object(s, start, limit=None):
    """Return the first balanced {...} object at/after `start`, or None. Ignores braces in strings.

    If `limit` is given, the object must START before it (so one call's args can't be taken from
    a later `<function=...>` opener's JSON).
    """
    i = s.find("{", start)
    if i == -1 or (limit is not None and i >= limit):
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
    matches = list(_OPEN_RE.finditer(completion))
    calls = []
    for idx, m in enumerate(matches):
        name = m.group(1).strip()
        # a call's args must start before the NEXT opener, so adjacent no-arg + arg calls
        # ('<function=a><function=b>{...}') don't hand b's JSON to a.
        limit = matches[idx + 1].start() if idx + 1 < len(matches) else None
        obj = _extract_json_object(completion, m.end(), limit)
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

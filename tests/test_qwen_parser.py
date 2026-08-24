"""Tests for the Qwen tool-call parser — the fix that reversed an earlier conclusion.

Run: python -m pytest tests/ -q   (or: python tests/test_qwen_parser.py)
No GPU, no agentdojo, no network — pure string parsing.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen_parser import extract_tool_calls  # noqa: E402


def test_qwen_bare_function_close():
    # the actual observed failure mode: closes with a bare <function>, not </function>
    calls = extract_tool_calls('<function=read_file>{"file_path": "landlord-notices.txt"}<function>')
    assert calls == [{"name": "read_file", "args": {"file_path": "landlord-notices.txt"}}]


def test_standard_function_close():
    calls = extract_tool_calls('<function=read_file>{"file_path": "x.txt"}</function>')
    assert calls == [{"name": "read_file", "args": {"file_path": "x.txt"}}]


def test_empty_args():
    # no-arg call, either close style -> empty dict, not a crash
    assert extract_tool_calls('<function=get_balance></function>') == [{"name": "get_balance", "args": {}}]
    assert extract_tool_calls('<function=get_balance><function>') == [{"name": "get_balance", "args": {}}]


def test_nested_json():
    calls = extract_tool_calls('<function=send>{"tx": {"to": "US123", "amt": 5}}<function>')
    assert calls == [{"name": "send", "args": {"tx": {"to": "US123", "amt": 5}}}]


def test_braces_inside_strings():
    # a } inside a string must not terminate the object early
    calls = extract_tool_calls('<function=note>{"msg": "a } b { c"}<function>')
    assert calls == [{"name": "note", "args": {"msg": "a } b { c"}}]


def test_multiple_tool_calls():
    calls = extract_tool_calls(
        '<function=a>{"x": 1}<function=b>{"y": 2}</function>')
    assert calls == [{"name": "a", "args": {"x": 1}}, {"name": "b", "args": {"y": 2}}]


def test_malformed_json():
    # broken JSON degrades to empty args, still returns the call (never raises)
    calls = extract_tool_calls('<function=send>{"to": "US123", oops}<function>')
    assert calls == [{"name": "send", "args": {}}]


def test_no_tool_call():
    assert extract_tool_calls("I need more information before I can help.") == []


def test_prose_before_call():
    calls = extract_tool_calls('Let me check that.\n<function=get_balance>{}<function>')
    assert calls == [{"name": "get_balance", "args": {}}]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print(f"\nall {len(fns)} parser tests passed")

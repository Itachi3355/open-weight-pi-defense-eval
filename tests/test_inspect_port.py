"""Inspect port: the error rule must match run_sweep.py exactly, and replays must reproduce it.

Run: python -m pytest tests/ -q   (skips if inspect-ai isn't installed)
"""
import glob
import os
import sys

import pytest

pytest.importorskip("inspect_ai")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inspect_ai.scorer import SampleScore, Score  # noqa: E402

import inspect_task as it  # noqa: E402


def _s(value, status):
    return SampleScore(score=Score(value=value, metadata={"status": status}), sample_id="x")


def test_errors_excluded_from_asr():
    # 1 cracked, 1 resisted, 2 errors -> ASR 1/2 (not 1/4); errors never count as defensive wins
    scores = [_s(1, "success"), _s(0, "success"), _s(0, "error"), _s(0, "error")]
    assert it.asr()(scores) == 0.5
    assert it.n_evaluable()(scores) == 2.0
    assert it.n_errors()(scores) == 2.0


def test_replays_match_run_sweep():
    files = sorted(glob.glob("results_isolated/*.jsonl"))
    if not files:
        pytest.skip("no recorded results to replay")
    for r in it.replay_all(files, log_dir=os.path.join("logs", "test_replay")):
        assert r["inspect_asr"] == pytest.approx(r["run_sweep_asr"], abs=1e-12), r["file"]
        assert (r["evaluable"], r["errors"]) == (r["expected_evaluable"], r["expected_errors"]), r["file"]

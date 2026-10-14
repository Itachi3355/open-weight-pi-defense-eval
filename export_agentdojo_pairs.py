#!/usr/bin/env python3
"""Export AgentDojo (user_task x injection_task) pairs to JSONL for the Inspect port.

One line per pair, in the same order run_sweep.py iterates them (user tasks outer, injection
tasks inner). Pure benchmark data -- AgentDojo's own task text and IDs, nothing generated.

    python export_agentdojo_pairs.py                      # banking -> agentdojo_banking_pairs.jsonl
    python export_agentdojo_pairs.py --suite slack --out agentdojo_slack_pairs.jsonl
"""
import argparse, json


def export(suite_name="banking", out=None, version="v1"):
    from agentdojo.task_suite.load_suites import get_suites
    suite = get_suites(version)[suite_name]
    out = out or f"agentdojo_{suite_name}_pairs.jsonl"
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for ut in suite.user_tasks.values():
            for it in suite.injection_tasks.values():
                f.write(json.dumps({
                    "id": f"{ut.ID}__{it.ID}",
                    "input": ut.PROMPT,
                    "metadata": {
                        "suite": suite_name,
                        "benchmark_version": version,
                        "user_task_id": ut.ID,
                        "injection_task_id": it.ID,
                        "injection_goal": it.GOAL,
                    },
                }) + "\n")
                n += 1
    print(f"wrote {n} pairs -> {out}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="banking")
    p.add_argument("--out", default=None)
    p.add_argument("--version", default="v1")
    a = p.parse_args()
    export(a.suite, a.out, a.version)

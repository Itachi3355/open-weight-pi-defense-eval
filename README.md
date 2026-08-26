# How Much of Open-Weight Prompt-Injection Defense Survives an Adaptive Attacker?

A reproducible, honest evaluation of black-box prompt-injection **defenses** on an
**open-weight** model, under both **static** and **adaptive** attacks, on the AgentDojo
banking suite with deterministic environment-state checks (no LLM judge).

Model: `Qwen/Qwen2.5-7B-Instruct` (fp16). Harness: `agentdojo==0.1.30` + modern vLLM.
See [`RESULTS.md`](RESULTS.md) for findings and caveats, and [`PAPER.md`](PAPER.md) for the
write-up. Parser fix is guarded by `tests/test_qwen_parser.py` (`python -m pytest tests/`).

![Classifier protection ratio (defended/undefended ASR) versus attacker scale: the shared-memory curve climbs 0.52, 0.70, 0.86, 0.89 toward the dashed 1.0 "defense erased" line, while the de-contaminated isolated-memory 32B point sits lower at 0.75.](figures/erosion_curve.svg)

*A defense doesn't have a robustness number — it has an erosion trend over attacker capability.
Under the original shared-memory protocol the classifier's protection ratio increased with
attacker scale (0.52 → 0.89); a corrected isolated 32B rerun still shows substantial erosion but
lower — ratio 0.75, not 0.89 — so the defense is eroded, not neutralized. (Only the 32B point is
de-contaminated so far; the clean slope below it is not yet measured.)*

## Headline — two findings (banking · `important_instructions` · adaptive K≤8)

Static bars (one-shot template): **none 18.75%**, **classifier 13.89%**. Adaptive ASR by
attacker scale, target fixed at Qwen2.5-7B:

| attacker | none | transformers_pi_detector | ratio (t/n) |
|---|---|---|---|
| 7B weak, K=4 | 15.97% | 8.33% | 0.52 |
| 7B strong, K=8 | 15.97% | 11.11% | 0.70 |
| 14B strong, K=8 | 14.58% | 12.50% | 0.86 |
| 32B strong, K=8 (shared winners) | 19.44% | 17.36% | 0.89 |
| **32B strong, K=8 (isolated winners)** | **21.68%** | **16.31%** | **0.75** |

1. **Capability threshold for adaptive dominance.** Attackers ≤14B stay *below* the static
   template; the 32B attacker exceeds it on **both** defenses — and this survives
   de-contamination (isolated winners: none 21.68% > 18.75%, classifier 16.31% > 13.89%).
   In this benchmark and attack setup, adaptive exceeded the static template only when the
   attacker substantially out-scaled the fixed 7B target.
2. **Erosion, not neutralization.** The classifier's protection ratio rises with attacker
   scale, but the clean isolated 32B point is **0.75** — it still cuts ASR ~25% against the
   strongest attacker. The shared-winner 0.89 was inflated by cross-condition payload
   transfer; isolating attacker memory per defense corrects it. See `RESULTS.md` for the
   full caveat (7B/14B not yet rerun isolated) and `results_isolated/` for raw transcripts.

## Reproduce

### Notebooks (Colab)
- `phase0_agentdojo_baseline.ipynb` — harness validation.
- `phase1_defense_matrix.ipynb` — static defense matrix.
- `phase2_adaptive_attacker.ipynb` — adaptive attacker.

Colab disconnects wipe files mid-run; prefer the headless script below for anything long.

### Headless (RunPod / Lambda / any SSH GPU box) — recommended

```bash
git clone https://github.com/Itachi3355/open-weight-pi-defense-eval && cd open-weight-pi-defense-eval
# Exact validated environments (two GPU driver stacks) are in requirements-lock.txt. Quick start:
pip install -U vllm && pip install "agentdojo==0.1.30" openai && pip uninstall -y torchaudio

tmux new -s sweep      # survives disconnects

# smoke test first (2 min): 1 user x 2 injections, K=2
python run_sweep.py --mode adaptive --defenses none --n-user 1 --n-inj 2 --K 2

# --- reproduce the reported experiments ---
# static baseline
python run_sweep.py --mode static   --defenses none transformers_pi_detector --git-push
# adaptive 7B baseline (weak, K=4)
python run_sweep.py --mode adaptive --defenses none transformers_pi_detector --K 4 --git-push
# strong 7B (richer red-team prompt + few-shot transfer of winning payloads + K=8)
python run_sweep.py --mode adaptive --defenses none transformers_pi_detector \
  --attacker-strength strong --K 8 --git-push

# attacker scaling: serve a larger attacker separately (e.g. on :18001) and point at it.
# 14B and 32B (AWQ) are the reported scaling points:
python run_sweep.py --mode adaptive --defenses none transformers_pi_detector \
  --attacker-strength strong --K 8 \
  --attacker-base-url http://localhost:18001/v1 \
  --attacker-model Qwen/Qwen2.5-32B-Instruct-AWQ --git-push

# isolated-winner protocol (de-contaminated: winners are NOT shared across defenses -- now the
# default). Reruns the 32B anchor into results_isolated/. Add --share-winners to reproduce the
# historical shared-pool rows instead.
python run_sweep.py --mode adaptive --defenses none transformers_pi_detector \
  --attacker-strength strong --K 8 --outdir results_isolated \
  --attacker-base-url http://localhost:18001/v1 \
  --attacker-model Qwen/Qwen2.5-32B-Instruct-AWQ --git-push
# Ctrl-b d to detach; `tmux attach -t sweep` to check in.
```

These commands reproduce the headline table above. The static + adaptive-7B/14B/32B and the
isolated 32B anchor have all been run; raw transcripts are in `results/` and `results_isolated/`.

`run_sweep.py` serves vLLM itself, applies the agentdojo↔vLLM compat shims in-process,
checkpoints **every pair** to `results/*.jsonl` (so a crash just resumes), prints ASR per
defense, and — with `--git-push` — commits results as it goes. On a 16GB T4 add
`--dtype float16` and an AWQ model (`--model Qwen/Qwen2.5-7B-Instruct-AWQ --quantization awq_marlin`).

## The compat shims (`patch_local.py` / `run_sweep.py`)

Three empirically-found fixes needed to run agentdojo 0.1.30 against modern vLLM + Qwen:
1. **content-part schema** — agentdojo sends `{"type":"text","content":…}`; vLLM wants `…"text":…`.
2. **int-digit cap** — some banking tool outputs exceed Python's 4300-digit str→int limit.
3. **tool-parser** — Qwen closes tool calls with a bare `<function>`, not `</function>`;
   brace-match the JSON instead. Without this, agents can't act and ASR/utility read false-low.

Plus a one-line fix to agentdojo's spotlighting defense (a self-referential lambda → infinite recursion).

## Next
1. **Isolated-winner reruns at 7B and 14B** — establish whether the erosion trend stays monotonic
   under the corrected (isolated) protocol; only the 32B anchor is de-contaminated so far.
2. **Repeated runs** — quantify serving-stack / sampling variance (the `none` 32B cell moved
   19.44% → 21.68% across stacks) with repeated full sweeps; bootstrap CI over tasks.
3. **Resolve the 14B → 32B transition** — intermediate attacker sizes to locate the threshold.
4. **Broaden** — clean static reruns for spotlighting/repeat_user_prompt; more targets/defenses;
   InjecAgent cross-check.
5. **White-box comparison** — GCG or another gradient/optimization attack vs the black-box LLM
   attacker. See `RESULTS.md`.

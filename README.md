# How Much of Open-Weight Prompt-Injection Defense Survives an Adaptive Attacker?

A reproducible, honest evaluation of black-box prompt-injection **defenses** on an
**open-weight** model, under both **static** and **adaptive** attacks, on the AgentDojo
banking suite with deterministic environment-state checks (no LLM judge).

Model: `Qwen/Qwen2.5-7B-Instruct` (fp16). Harness: `agentdojo==0.1.30` + modern vLLM.
See [`RESULTS.md`](RESULTS.md) for findings and caveats, and [`PAPER.md`](PAPER.md) for the
write-up. Parser fix is guarded by `tests/test_qwen_parser.py` (`python -m pytest tests/`).

## Current headline (clean 2×2, banking · important_instructions · K=4)

| defense | clean static ASR | adaptive ASR (K=4) |
|---|---|---|
| none | 18.75% | 15.97% |
| transformers_pi_detector | 13.89% | 8.33% |

**Honest status:** with a *weak* attacker (same 7B, simple prompt, K=4), adaptive ASR is
*below* static for both conditions — the attacker does not yet beat the hand-crafted
`important_instructions` template. The adaptive-beats-static claim is **not demonstrated
yet**; a stronger attacker is the next required experiment. (An earlier "adaptive doubles
static" reading was a contamination artifact — clean-adaptive vs pre-parser-fix static —
and has been retracted in `RESULTS.md`.)

## Reproduce

### Notebooks (Colab)
- `phase0_agentdojo_baseline.ipynb` — harness validation.
- `phase1_defense_matrix.ipynb` — static defense matrix.
- `phase2_adaptive_attacker.ipynb` — adaptive attacker.

Colab disconnects wipe files mid-run; prefer the headless script below for anything long.

### Headless (RunPod / Lambda / any SSH GPU box) — recommended

```bash
git clone https://github.com/Itachi3355/open-weight-pi-defense-eval && cd open-weight-pi-defense-eval
pip install -U vllm && pip install "agentdojo==0.1.30" openai && pip uninstall -y torchaudio

tmux new -s sweep      # survives disconnects

# smoke test first (2 min): 1 user x 2 injections, K=2
python run_sweep.py --mode adaptive --defenses none --n-user 1 --n-inj 2 --K 2

# clean static + weak-attacker adaptive baselines
python run_sweep.py --mode static   --defenses none transformers_pi_detector --git-push
python run_sweep.py --mode adaptive --defenses none transformers_pi_detector --K 4 --git-push

# STRONGER attacker (the experiment that decides the thesis): richer prompt + few-shot
# transfer of winning payloads + higher K. On a 40GB A100 you can serve a larger attacker
# model separately and point --attacker-base-url/--attacker-model at it.
python run_sweep.py --mode adaptive --defenses none transformers_pi_detector \
  --attacker-strength strong --K 8 --git-push
# Ctrl-b d to detach; `tmux attach -t sweep` to check in.
```

**Decision gate:** if `--attacker-strength strong` pushes adaptive ASR **above** the clean
static bar (>18.75% on none, >13.89% on the classifier), the adaptive-beats-static thesis
holds — scale to the model matrix. If it stays below, that is itself the honest finding
(these defenses hold better than the doom literature implies on an open 7B / black-box
LLM attacker) — reframe accordingly. Run this ONE experiment before building breadth.

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
Stronger attacker (few-shot winning payloads, higher K, larger/separate attacker model) →
clean static reruns for spotlighting/repeat_user_prompt → InjecAgent cross-check → model
matrix → variance/CI. See `RESULTS.md`.

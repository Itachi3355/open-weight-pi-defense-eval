# CLAUDE.md — Prompt-Injection Defense Evaluation Paper

Persistent project context for Claude Code. Read this first every session.

## What this project is

An independent empirical research paper evaluating prompt-injection **defenses** on
**open-weight models**, under both static and *adaptive* attacks. The goal is a
credible, reproducible, arXiv-quality paper — the kind of self-contained empirical
work that strengthens an Anthropic Fellows application. It is NOT a submission to
Anthropic; there is no "Anthropic accepts my paper" pathway. The paper stands on its
own (arXiv + a workshop), and the research ability it demonstrates is what helps the
application.

## The one non-negotiable principle

**Adaptive evaluation is what makes or breaks this paper.** A paper reporting only
*static* attack-success numbers is treated as already-refuted — this is the entire
thrust of the 2025 consensus paper (Nasr, Carlini et al., "The Attacker Moves
Second," arXiv:2510.09023), which bypassed 12 recent defenses at >90% ASR even though
most originally reported near-zero. Every phase must build toward an adaptive attacker.
Static-only results are a dismissed paper.

## Recommended paper concept

"How Much of Open-Weight Prompt-Injection Defense Survives an Adaptive Attacker?
A Reproducible Evaluation."

Take established benchmarks (AgentDojo primary, InjecAgent cross-check), implement a
matrix of black-box defenses, and evaluate them across a matrix of open-weight models
under BOTH static and adaptive attacks. Report attack-success-rate (ASR) and
utility-under-attack. The contribution is an honest, open-weight quantification of how
much reported defense robustness is an artifact of non-adaptive evaluation — plus a
reusable harness.

Alternate concepts if the main one gets blocked or over-crowded:
- Does reasoning help? (Qwen3 thinking on/off, DeepSeek-R1 distills) — under-studied.
- Defense-stacking / defense-in-depth: is combining defenses more-than-additive, or
  just over-defense + utility loss?
- Over-defense / false-positive characterization: the security–utility–latency cost
  of each defense on benign-but-instruction-like data.

## Compute setup

- **No usable local GPU.** Dev machine is Intel UHD 620 integrated graphics — no CUDA.
  Do NOT assume local model inference on the dev box.
- **Google Colab Pro** is the compute environment. Priority A100 (40GB) / L4 (24GB) /
  T4 (16GB), longer runtimes, background execution. Watch the compute-unit budget —
  do not burn units on unvalidated pipelines (that's what Phase 0 is for).
- Code is authored/version-controlled locally in `C:\Users\mural\VSCODE_Fls\ANTHROPIC\`;
  runs execute in Colab. Notebooks are the interface between the two.
- Hugging Face account + access token needed for weights. Llama models require a
  one-time license click-through on their HF page.

## Model matrix (open weights, Colab-runnable)

- Llama-3.1-8B-Instruct (primary)
- Qwen3-8B (test thinking on/off)
- Gemma-3-4B
- Mistral-7B-Instruct
- Meta-SecAlign-8B (model-level-defense baseline)
- Llama-3.3-70B (only if renting a bigger GPU; optional)
- Temperature 0 for reproducibility. 3 seeds for variance.

## Defense matrix

(i) none · (ii) delimiting · (iii) sandwiching / repeat-user-prompt · (iv) spotlighting
(datamarking + base64) · (v) fine-tuned classifier (ProtectAI deberta / PromptGuard,
the transformers PI detector bundled with AgentDojo) · (vi) tool-filter · (vii)
model-level SecAlign as a comparison point.

## Attack matrix

- Static: AgentDojo built-in attacks (ignore-previous, important-instructions,
  system-message). "important instructions" is the strong default.
- Adaptive: an AutoDojo-style black-box attacker (LLM-in-the-loop, local Qwen as
  attacker LLM, ASR fed back to optimize). This is the crux — see the non-negotiable.
- Optional white-box: one GCG run on the smallest model for a single data point.

## Metrics

Targeted ASR (static and adaptive) · clean utility · utility-under-attack ·
over-defense/false-positive rate on benign instruction-like content. Mean ± CI over 3
trials. Break results out by task-specification precision (fully-specified vs
action-open) — AutoDojo shows action-open tasks are far more vulnerable (28% overall
vs 64% on action-open against a filter that had reduced static ASR to 0%).

## Phased plan

**Phase 0 — prove the pipeline (do this first, one session).**
Stand up AgentDojo in Colab, load ONE small model, run the banking suite with the
"important instructions" attack and NO defense, and reproduce a ballpark ASR matching
published AgentDojo numbers. If the number is in range, the harness is trustworthy. If
not, fix it before building anything. Use deterministic environment-state checks — NOT
an LLM judge — for success.

**Phase 1 — the experiment.**
Add the defense matrix + adaptive attacker, expand to the full model matrix, collect
ASR + utility-under-attack with 3 seeds. Release code, configs, logs.

**Phase 2 — write-up.**
arXiv preprint (cs.CR / cs.CL) first. Then a workshop. Honest, reproducible,
negative-result-friendly. Over-claiming a new "robust" defense is the main failure
mode to avoid.

## Target venues (2026–2027)

- **arXiv (cs.CR / cs.CL)** — default first step, post regardless.
- **IEEE SaTML 2027** — strongest realistic archival target. Deadline ~Sept 29, 2026.
- **NeurIPS 2026 FLMSec workshop** — best topical fit but deadline ~Aug 22, 2026
  (essentially now; only if a draft already exists). Non-archival.
- Other NeurIPS 2026 workshops (~Aug 29 deadlines, non-archival): "Who Verifies the
  Agents?", "Trustworthy AI for Good".
- **LessWrong / Alignment Forum** — post alongside arXiv; read by Anthropic researchers.
- Verify all deadlines/page limits on the primary CFP before committing.

## Key references (verify headline numbers against primaries before citing)

- Greshake et al. 2023 — indirect prompt injection (arXiv:2302.12173)
- Zverev et al. — SEP / instruction-data separation (ICLR 2025, arXiv:2403.06833)
- InjecAgent (Zhan et al., ACL Findings 2024, arXiv:2403.02691)
- AgentDojo (Debenedetti et al., NeurIPS 2024 D&B, arXiv:2406.13352) — the harness
- Spotlighting (Hines et al., Microsoft, arXiv:2403.14720)
- StruQ (USENIX Sec 2025); SecAlign (CCS 2025); Meta-SecAlign (arXiv:2507.02735)
- CaMeL "Defeating Prompt Injections by Design" (Google, arXiv:2503.18813)
- "The Attacker Moves Second" (Nasr, Carlini et al., arXiv:2510.09023) — THE motivation
- AutoDojo — adaptive black-box extension of AgentDojo (arXiv:2606.15057)
- OWASP LLM Top 10 — prompt injection is LLM01:2025

## Related existing work by this author

Two shipped Agent Skills (public, MIT) that this paper's tooling connects to:
- `verify-ai-output` — epistemic/hallucination auditing
- `prompt-injection-audit` — injection red-teaming methodology
GitHub: github.com/Itachi3355 (git identity: Itachi3355,
154012933+Itachi3355@users.noreply.github.com)

## First Claude Code prompt to run

"Read CLAUDE.md. We're at Phase 0 — help me build the AgentDojo baseline notebook for
Colab Pro: clone AgentDojo, load Llama-3.1-8B (or Qwen if Llama license isn't cleared
yet), run the banking suite with the important-instructions attack and no defense, and
print the ASR so we can check it against the published number. Deterministic success
checks only."

## Working style notes

- Direct, detailed, actionable outputs. Full artifacts over high-level guidance.
- Honest fit assessments over reassurance. Flag when something won't work.
- Reproduce a known baseline before building on top of any harness.

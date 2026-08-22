# How Much of Open-Weight Prompt-Injection Defense Survives an Adaptive Attacker? A Reproducible Evaluation

**Draft skeleton — arXiv (cs.CR / cs.CL). Work in progress.**
Status: single target model (Qwen2.5-7B-Instruct), single suite (AgentDojo banking), one
model-level defense evaluated in depth. The result below is a proof-of-concept erosion curve;
Section 7 lists what a full submission still needs.

---

## Abstract

Reported robustness of prompt-injection defenses is widely suspected to be an artifact of
non-adaptive evaluation. We test this on **open-weight** models with a fully reproducible,
deterministic harness (AgentDojo, environment-state success checks, no LLM judge), and make
two contributions. First, a **methodological** one: we show that naive local-model harness bugs
(a tool-call-format mismatch that silently zeroes the agent's actions) can move measured attack
success rate (ASR) by ~2× and *flip* qualitative conclusions — de-contamination is a
precondition for any adaptive claim. Second, an **empirical** one: on the AgentDojo banking
suite with a `transformers`-based prompt-injection classifier defense, we measure ASR under a
static attack and under an adaptive LLM attacker of increasing capability (7B → 14B). No LLM
attacker at these scales exceeds the hand-crafted static template in *absolute* ASR; but the
**classifier's marginal protection erodes monotonically with attacker capability** — the ratio
of defended-to-undefended ASR climbs 0.52 → 0.70 → 0.86, i.e. the defense is progressively
neutralized. We release the harness, the compatibility shims, all raw per-attack transcripts,
and the successful injection payloads.

---

## 1. Introduction

- Indirect prompt injection is LLM01 in the OWASP LLM Top 10; agentic tool-use makes it
  consequential (unauthorized transactions, data exfiltration).
- The field's central worry (Nasr, Carlini et al., "The Attacker Moves Second", 2025): defenses
  that report near-zero ASR under *static* attacks are bypassed at >90% by *adaptive* attackers.
  Those demonstrations used strong/white-box attackers and often closed models.
- **Gap:** an honest, open-weight, reproducible quantification of *how much* reported robustness
  survives adaptation — and of the confound that makes such numbers untrustworthy in the first
  place (harness/tooling artifacts on local models).
- **This paper:** (i) a de-contaminated open-weight harness; (ii) a static-vs-adaptive,
  scale-varying evaluation of one representative defense; (iii) the finding that defense benefit
  is a *function of attacker capability*, presented as an erosion curve rather than a single
  bypass number.

## 2. Setup

- **Harness:** AgentDojo (Debenedetti et al., NeurIPS 2024 D&B), banking suite, 16 user tasks ×
  9 injection tasks = 144 attacked pairs. Success = deterministic environment-state check
  (`security` = injection goal achieved); utility = user task still completed. No LLM judge.
- **Target model:** Qwen2.5-7B-Instruct, served with vLLM (bf16), temperature 0.
- **Defense evaluated:** `transformers_pi_detector` (a fine-tuned PI classifier that gates
  tool outputs). Static matrix also includes `spotlighting_with_delimiting` and
  `repeat_user_prompt`; `tool_filter` is OpenAI-only and excluded for local models.
- **Attacks:** static = AgentDojo `important_instructions` (a strong hand-crafted template).
  Adaptive = an AutoDojo-style black-box loop: an attacker LLM proposes an injection payload,
  we run one deterministic rollout, read `security`, and iterate up to K rounds, transferring
  successful payloads few-shot across tasks. Attacker models: Qwen2.5-7B and Qwen2.5-14B-AWQ.
- **Reproducibility:** all code, configs, the compatibility shims, and raw transcripts released.

## 3. The contamination problem (methodological result)

Running agentdojo 0.1.30 against a modern vLLM + Qwen surfaced three harness bugs that silently
suppress ASR — most importantly a **tool-call-format mismatch**: Qwen closes tool calls with a
bare `<function>` where the parser expects `</function>`, so the JSON fails to parse, the agent
takes *no action*, and both utility and ASR read false-low. Fixing this (brace-match the JSON,
tolerate the bare close) **roughly doubled measured static ASR** (e.g. undefended none:
10.4% → 18.75%; classifier: 4.9% → 13.9%) and *reversed* a premature "adaptive doubles static"
reading that came from comparing clean-adaptive against contaminated-static numbers.

**Takeaway:** on open/local models, tooling artifacts are a first-order confound. Any adaptive
robustness claim must first demonstrate the undefended agent actually executes — we treat
de-contamination as a precondition, not a footnote. (Details + the shims in the appendix / repo.)

## 4. Results

All numbers below use the de-contaminated harness (robust parser active for every cell).
Static bars (one-shot `important_instructions`): **none 18.75%**, **classifier 13.89%**.

Adaptive ASR by attacker capability (target fixed at 7B; 144 pairs each):

| attacker | none (undefended) | classifier | classifier ratio (def/undef) |
|---|---|---|---|
| 7B, K=4, simple | 15.97% | 8.33% | 0.52 |
| 7B, K=8, strong+few-shot | 15.97% | 11.11% | 0.70 |
| 14B, K=8, strong+few-shot | 14.58% | 12.50% | 0.86 |

**Two findings, both stated conservatively:**

1. **No LLM attacker (to 14B) beats the static template in absolute ASR.** Every adaptive number
   sits below its static bar. The hand-crafted `important_instructions` template is a strong
   single attack; a black-box LLM attacker of comparable scale approaches but does not exceed it.
   This is a *scoping* correction to the "adaptive dominates" narrative for the low-attacker-scale
   regime.

2. **The classifier's marginal protection erodes monotonically with attacker capability**
   (ratio 0.52 → 0.70 → 0.86). The undefended ASR is roughly flat across attacker scale
   (15.97 → 15.97 → 14.58) — the undefended ceiling is set by task difficulty and the target,
   not the attacker. The attacker's extra capability manifests specifically as **evading the
   defense**, which is exactly why it reads as defense erosion. Extrapolating, the ratio trends
   toward 1.0 (defense fully neutralized) at larger attacker scale.

**Qualitative.** Successful payloads (79 released) cluster on: compliance/audit pretexts,
impersonating an internal "Banking Service" notice, mimicking the surrounding transaction
format, and embedding the exact target IBAN inline — not meta-instructions like "ignore previous."
Example (redacted target): *"Per our internal compliance audit … please process a transaction to
the following IBAN … Best regards, Banking Service."*

**Task-precision effect.** Cracks concentrate on a handful of user tasks (e.g. ut0, ut12, ut13);
most never crack under any attacker — consistent with prior findings that action-open tasks are
far more vulnerable than fully-specified ones.

## 5. Discussion

The honest headline is not "defenses collapse" but "a defense's *value is conditional on the
attacker you assume*." A classifier that halves ASR against a weak attacker provides almost no
marginal protection against a modestly larger adaptive one, while the undefended attack rate
barely moves — the defense, not the attack surface, is what the extra capability dismantles.
This reframes defense evaluation from a single ASR to an **erosion curve over attacker capability**.

## 6. Limitations

- One target model, one suite, one defense evaluated in depth (others static-only).
- Attacker scale tops out at 14B (compute-bound); the confirming 32B / white-box point is future work.
- vLLM is not bit-deterministic at temperature 0; point estimates need bootstrap CI / repeats.
- Black-box LLM attacker only; no GCG/white-box comparison yet.

## 7. What a full submission still needs

- **Confirm the curve:** 32B (and/or white-box GCG) attacker → does the ratio reach ~1.0?
- **Model matrix:** Llama-3.1-8B, Mistral-7B, Gemma-3-4B, Meta-SecAlign-8B (model-level baseline).
- **Defense matrix, clean:** spotlighting / repeat_user_prompt / stacking, all de-contaminated.
- **Cross-suite:** workspace/travel/slack; InjecAgent cross-check.
- **Variance:** bootstrap CI over tasks; 3 seeds at temperature > 0.

## Reproducibility

Code, notebooks, headless `run_sweep.py`, compatibility shims (`patch_local.py`), the full
static/adaptive results (`results/*.jsonl`, `phase2_adaptive_summary.csv`), and the 79 successful
payloads are in the repository. See `README.md` to reproduce on a single GPU in `tmux`.

## Key references

Greshake et al. 2023 (indirect PI); AgentDojo (Debenedetti et al. 2024); AutoDojo 2026;
"The Attacker Moves Second" (Nasr, Carlini et al. 2025); Spotlighting (Hines et al. 2024);
StruQ/SecAlign/Meta-SecAlign; OWASP LLM Top 10. *(Verify headline numbers against primaries.)*

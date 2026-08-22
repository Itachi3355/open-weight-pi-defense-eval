# A prompt-injection defense doesn't have one robustness number — it has a curve

*Open-weight, reproducible, and with a cautionary tale about measurement in the middle.*

**TL;DR.** I evaluated a prompt-injection classifier defense on an open 7B agent (AgentDojo
banking, deterministic checks, no LLM judge) under a static attack and an adaptive LLM attacker
of growing size. Two things came out. (1) A boring harness bug silently zeroed the agent's
actions and *halved* every attack-success number — fixing it flipped a conclusion I'd already
written down. (2) Once clean, the interesting result isn't "the defense collapses" — it's that
**the defense's benefit shrinks smoothly as the attacker gets stronger**: the ratio of
defended-to-undefended attack success climbs 0.52 → 0.70 → 0.86 as the attacker goes 7B → 7B+ →
14B. A defense doesn't have a robustness number; it has an erosion curve over attacker capability.

## Why I did this

The 2025 "Attacker Moves Second" paper is the field's cold shower: defenses reporting near-zero
attack-success-rate (ASR) under static attacks get bypassed >90% by adaptive ones. But those
demos leaned on strong or white-box attackers, often on closed models. I wanted the honest
open-weight version: take a real defense, a reproducible harness with *deterministic* success
checks (not an LLM judge grading itself), and ask how much robustness actually survives — using
attackers you or I could run.

## The part nobody warns you about: your harness is lying to you

Before any "adaptive" story, you have to trust the numbers. I didn't, and I was right not to.

Running AgentDojo against a current vLLM + Qwen, the agent looked *impressively* robust —
suspiciously robust. The cause: Qwen closes tool calls with a bare `<function>` tag where the
harness's parser expected `</function>`. The JSON failed to parse, so the agent **took no
action at all** — and an agent that does nothing never completes the task *and* never falls for
the injection. Both utility and ASR read false-low. It looks like safety. It's a parser bug.

Fixing it (brace-match the JSON, tolerate the weird close tag) roughly **doubled** measured
static ASR — undefended 10% → 19%, classifier 5% → 14% — and *reversed* a conclusion I'd
tentatively written ("adaptive beats static!"), which turned out to be an artifact of comparing
clean adaptive numbers against still-contaminated static ones.

Lesson worth the whole project: **on local models, tooling artifacts are a first-order confound.**
If your undefended agent isn't actually executing, you're measuring your parser, not your defense.

## The actual result

Clean harness. Static attack bar: undefended 18.75%, classifier 13.89%. Then I ran an
AutoDojo-style adaptive attacker — an LLM proposes an injection, I run one deterministic rollout,
read success, iterate, and transfer winning payloads across tasks — at increasing attacker size.

| attacker | undefended | classifier | ratio (def / undef) |
|---|---|---|---|
| 7B, weak | 15.97% | 8.33% | **0.52** |
| 7B, strong + few-shot, K=8 | 15.97% | 11.11% | **0.70** |
| 14B, strong + few-shot, K=8 | 14.58% | 12.50% | **0.86** |

Two honest reads:

- **No LLM attacker I ran (up to 14B) beat the hand-crafted static template in absolute ASR.**
  So "adaptive dominance" isn't the story at these scales — the crafted template is just a strong
  attack, and a same-ish-scale LLM attacker approaches but doesn't pass it.
- **But the classifier is being neutralized.** Against a weak attacker it roughly halves ASR
  (ratio 0.52). Against a 14B attacker it barely helps (0.86). The undefended rate is basically
  flat across attacker size — so the extra attacker capability isn't raising the raw attack rate,
  it's specifically *learning to evade the defense*. That's the erosion.

The winning payloads are mundane and human: compliance-audit pretexts, fake "Banking Service"
notices, format-mimicry, the target account number embedded inline. No "ignore previous
instructions." The stuff that works looks like legitimate in-band content.

## The reframe

A defense evaluated at one attacker strength gives you one number, and that number is close to
meaningless on its own. The same classifier is "halves attacks" or "does almost nothing"
depending entirely on who's attacking. The useful object isn't ASR — it's the **erosion curve**:
defended benefit as a function of attacker capability. Mine has three points and trends toward
"no protection." I ran out of GPU credits before the 32B point that would (I bet) push it to ~1.0.

## Caveats (it's a proof of concept)

One target model, one suite, one defense in depth, attacker capped at 14B, no white-box
comparison, and vLLM isn't bit-deterministic so these want confidence intervals. It's a
skeleton, not a finished paper. But the methodology (de-contaminate first, then measure the
curve) and the framing (robustness is conditional, plot it) are the transferable parts.

## Everything's open

Harness, the compatibility shims, the headless runner, all raw per-attack transcripts, and the
79 successful payloads: **github.com/Itachi3355/open-weight-pi-defense-eval**. Reproduces on a
single GPU in `tmux`. If you have an 80GB card and ten minutes, the 32B point is one command —
I'd love to see whether the curve hits 1.0.

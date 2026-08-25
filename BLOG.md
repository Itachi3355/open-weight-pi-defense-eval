# A prompt-injection defense doesn't have one robustness number — it has a curve

*Open-weight, reproducible, and with a cautionary tale about measurement in the middle.*

**TL;DR.** I evaluated a prompt-injection classifier defense on an open 7B agent (AgentDojo
banking, deterministic checks, no LLM judge) under a static attack and an adaptive LLM attacker
of growing size (7B → 14B → 32B). Three things came out. (1) A boring harness bug silently
zeroed the agent's actions and *halved* every attack-success number — fixing it flipped a
conclusion I'd already written down. (2) **A defense doesn't have a robustness number; it has an
erosion curve** — the ratio of defended-to-undefended attack success rises as the attacker
scales up, but the defense is eroded, not neutralized: at the cleanest measured point the 32B
adaptive attacker still gets its ASR cut ~25% by the classifier (ratio 0.75). (3) **Adaptive
dominance has a capability threshold**: attackers up to 14B *lose* to a hand-crafted static
template, but a 32B attacker *beats* it on both the undefended and defended agent. "Adaptive
beats static" is true — but only once the attacker out-scales the target.

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

| attacker | undefended | classifier | ratio (def / undef) | attacker memory |
|---|---|---|---|---|
| 7B, weak | 15.97% | 8.33% | **0.52** | shared |
| 7B, strong + few-shot, K=8 | 15.97% | 11.11% | **0.70** | shared |
| 14B, strong + few-shot, K=8 | 14.58% | 12.50% | **0.86** | shared |
| 32B, strong + few-shot, K=8 | 19.44% | 17.36% | **0.89** | shared |
| **32B, strong + few-shot, K=8** | **21.68%** | **16.31%** | **0.75** | **isolated** |

Static bars for comparison: undefended **18.75%**, classifier **13.89%**.

That last row is the one I trust. The "shared" rows let the classifier's attacker reuse
payloads it found against the *undefended* agent — which flatters the attacker and makes the
defense look more neutralized than it is. When I gave each defense its own isolated attacker
memory and reran the 32B point, the classifier's ASR dropped (17.4 → 16.3%) and the ratio fell
0.89 → **0.75**. Same lesson, smaller: the defense erodes, it doesn't evaporate.

Two reads:

- **Adaptive dominance kicks in at 32B — and it's not an artifact.** The 7B and 14B attackers
  stay *below* the static template; at those scales the crafted template just wins. The 32B
  attacker beats it on both, and this survives the clean isolated rerun: undefended
  21.7% > 18.8%, classifier-defended 16.3% > 13.9%. Adaptive wins *once the attacker out-scales
  the target*. There's a threshold, and I walked across it between 14B and 32B.
- **The classifier is eroded, not neutralized.** Against a weak attacker it roughly halves ASR
  (ratio 0.52). Against the strongest attacker, cleanly measured, it still cuts ASR ~25%
  (ratio 0.75). The extra attacker capability is spent partly on *learning to evade the defense* —
  that's the erosion — but the defense keeps real value even at the top of my range.

The winning payloads are mundane and human: compliance-audit pretexts, fake "Banking Service"
notices, format-mimicry, the target account number embedded inline. No "ignore previous
instructions." The stuff that works looks like legitimate in-band content.

## The reframe

A defense evaluated at one attacker strength gives you one number, and that number is close to
meaningless on its own. The same classifier "halves attacks" or "cuts them a quarter"
depending entirely on who's attacking. The useful object isn't ASR — it's the **erosion curve**:
defended benefit as a function of attacker capability. Mine runs 0.52 → 0.75 (clean at the top
point) and, more strikingly, has a **threshold**: somewhere between a 14B and a 32B attacker,
adaptive attacks go
from losing to a hand-crafted template to beating it. If you evaluate your defense against a
7B attacker and ship it, you haven't measured its robustness — you've measured that particular
matchup. Someone with a bigger attacker gets a different, worse answer.

## Caveats (it's a proof of concept)

One target model, one suite, one defense in depth, attacker capped at 32B, no white-box
comparison, and vLLM isn't bit-deterministic so these want confidence intervals (the `none`
column even ticks *down* 16→14.6% before jumping to 19.4% at 32B — noise you'd smooth out with
repeats). The exact threshold between 14B and 32B is unmeasured. It's a skeleton, not a finished
paper. But the methodology (de-contaminate first, then measure the curve) and the framing
(robustness is conditional — plot it against attacker scale) are the transferable parts.

## Everything's open

Harness, the compatibility shims, the headless runner, all raw per-attack transcripts, and the
successful payloads: **github.com/Itachi3355/open-weight-pi-defense-eval**. Reproduces on a
single GPU in `tmux` (the 32B run wants an 80GB card — A100/H100). The obvious next experiment
is to fill in the 14B→32B gap and find exactly where adaptive crosses static — one `run_sweep.py`
invocation per attacker size. I'd love to see someone map that threshold across defenses.

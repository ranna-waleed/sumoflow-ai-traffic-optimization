# DQN diagnostic false-valid correction

Date: 2026-10-27  
Branch: `research/ieee-reproducibility`

## Preserved historical diagnostic

The run at
`DeepQN/diagnostics/phase3_synthetic_night_v1/run-20261027T074227Z`
is preserved byte-for-byte. Its report incorrectly records
`diagnostic_behavior_valid=true` because the former reader discarded the message-log
summary line `Emergency Braking: 1`. This run and its seven checkpoints are therefore
**behavior-invalid historical exploratory evidence** and must not be selected as a
valid diagnostic or research checkpoint.

- Report SHA-256: `23f1a347eb9c9c380beb049edd9b57bc3fddbc7d145b3761bc0a1b944081dcad`
- Decision telemetry SHA-256: `244360972b7ad3be64f2eaf23b3610da97e17bbda136376604035590286c3d6c`
- Message log SHA-256: `4c2aa1fc6c22bf01cad7243e8de320cf493a07ba2882bf8bddb78613b061eb34`

No existing run file or checkpoint was modified or deleted.

## Correction

New diagnostics configure SUMO's supported `--error-log` in addition to the existing
message log. Error, message, and captured process logs are required and reconciled.
Detailed adverse-event lines are retained with their contributing source names and are
deduplicated by normalized content. `Emergency Braking: N` is parsed independently
from the message-log final counters, including when no detailed warning line exists.
Missing, unreadable, or contradictory telemetry fails closed. A behavior-invalid run
retains optimizer/loss telemetry but does not call checkpoint saving.

## Physical-trace assessment

At decision index 49 (`sim_time=79700`), TLS `96621100` accepted a requested switch
from controller phase 0. For every recorded one-second interval from 79690 through
79700, the physical boundary trace reports stable phase 2, state `GGGGGggggg`; the
warning time 79693 falls within that stable-green interval. The braking vehicle was on
internal lane `:96621100_0_3`, link index 3, connecting lane 3 of `690516091#1` to lane
3 of `10873186#9`. Link 3 was protected green (`G`), while links 5-9 from
`10873186#8` were permissive green (`g`).

The evidence does not show a yellow/red transition at the warning instant, so it does
not support attributing this event to a mid-interval signal transition. A downstream
interaction or merge interaction is plausible, but vehicle trajectory/gap/leader data
were not recorded, so causal attribution is unresolved.

No simulation or training was run for this correction.

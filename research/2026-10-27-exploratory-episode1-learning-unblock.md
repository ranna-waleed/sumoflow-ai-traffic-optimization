# Exploratory episode-1 learning unblock — 2026-10-27

The immutable run `night-v6-stopturn-r1-3ep-seed41001` completed its first
7,200-second episode but was labeled `failed_integrity`. Reclassification of
its preserved report shows the sole integrity failure came from two
`performs emergency stop ... because of a red traffic light` warnings. These
are adverse traffic behavior, like emergency braking, rather than malformed
input or failed execution. The episode remains `nonpublication_behavior_flagged`
because it contains three emergency-braking details, two emergency stops and
one jam teleport. No historical report or progress file was changed.

The teleporting taxi `flow_648.1` followed a valid route and waited on internal
lane `:3105658889_0_2`, the permitted straight connection from
`10873186#7_5` to `10873186#8_2`. Junction `3105658889` is an uncontrolled
priority junction immediately upstream of the 0.2 m approach to controlled
TLS `96621100`. The event is consistent with downstream queue spillback. The
run did not persist decision-level action or physical-phase traces, so it does
not establish whether the DQN schedule caused the queue.

The warning taxonomy now records `emergency_stops` separately, permits them in
execution integrity, and retains them in the behavior-review classification.
Route errors, stop errors, unsafe signals, missing telemetry, malformed
metrics, demand mismatch, and incomplete horizons still fail integrity.

An integrity-valid behavior-flagged episode is atomically snapshotted before
the trainer stops for collision/teleport review. Resume remains blocked unless
the operator supplies `--acknowledge-traffic-review`; the acknowledgment is
recorded as nonpublication and explicitly is not safety approval. This lets a
reviewed 720-transition replay state continue, reaching the 1,000-transition
training threshold 280 decisions into episode 2. Failed-integrity runs remain
non-resumable.

Direct import of native fixed-time trajectories was not implemented: native
phase progression lacks the DQN binary requested-action labels, so doing so
would fabricate replay actions unless a dedicated behavior-policy adapter is
first built and validated.

Focused result: 15 tests passed in 4.277 seconds. Compilation and targeted
diff checks passed. No SUMO simulation or training was executed.

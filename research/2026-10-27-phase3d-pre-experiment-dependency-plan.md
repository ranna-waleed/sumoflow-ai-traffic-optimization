# Phase 3.0D pre-experiment dependency plan

This plan separates prerequisites from outputs so absent future checkpoints or results do not
incorrectly block the stage that is meant to produce them. Every stage remains independently
approval-gated. No stage is authorized by this document.

## Gate 0 — close versioned inputs

Required before any scientific model fitting:

- Commit/review an immutable five-pool demand registry with explicit route bytes, canonical
  physical fingerprints, generation seeds, source/config/network hashes and profile windows.
- Resolve whether the study is transparently synthetic or field-calibrated. Without external
  evidence it must be labeled synthetic/uncalibrated.
- Decide the exact Tahrir detector mapping scope. External approval is required for a
  traffic-engineering claim; otherwise adopt a bounded synthetic geometry and say so.
- Freeze the effective SUMO configuration, every additional file, SUMO 1.26 build, step length,
  waiting memory, teleport/collision settings and output options.
- Freeze the seed-allocation contract. Policy replicate indices must map explicitly to DQN RNG
  seeds; forecast seeds must be allocated; the learned-arm training-seed pairing/crossing rule
  must no longer be null.

Exit evidence: externally referenced hashes or an explicitly limited synthetic protocol,
non-null registry/config identities, no held-out overlap, and a reviewed diff. No model output is
required at this gate.

## Gate 1 — bounded training-only diagnostic pilot

Purpose: estimate runtime/memory and verify the orchestration/resume path, not choose performance.

Inputs: one training-pool realization, one declared diagnostic DQN seed, calibration references
from training/calibration only, both observation arms' effective configurations, strict resource
and wall-clock limits, and no validation/test/evaluation access.

Outputs: resource ledger, failure diagnostics, resume equivalence evidence and a proposed fixed
training budget. Pilot weights are diagnostic and cannot become publication checkpoints.

## Gate 2 — forecast dataset and model production

Inputs: approved-scope detector mapping, training/validation/forecast-test demands, event/source
hashes, one-second observations, 60-to-30 causal contract, training-only scaler, allocated
forecast seeds and preregistered selection rule.

Outputs: dataset/scaler manifests; per-seed nonnegative BiLSTM checkpoints; validation-selected
checkpoint identity; held-out forecast-test scores versus zero, persistence, historical mean and
moving average. Controller-evaluation data is prohibited.

Exit gate: strict adapter loads the exact selected checkpoint and mapping; no fallback; likelihood
and count calibration are reported. The checkpoint is an output of this stage, not an input.

## Gate 3 — DQN training production

Inputs: frozen budget from Gate 1, four-profile reward-calibration artifact, five mapped DQN RNG
seeds per arm, frozen training/validation schedules, observation v3, interval reward v1, strict
no-forecast and selected verified-forecast artifacts, and complete interruption-equivalent
snapshot contracts.

Outputs: five complete research ensembles per arm, checkpoint manifests/hashes, learning curves,
resource/failure records and validation-only checkpoint selection. No controller-evaluation
demand is accessed.

Exit gate: every ensemble loads atomically under its exact architecture/config/run manifest;
policy replicate index and actual RNG seed are both recorded.

## Gate 4 — held-out controller evaluation

Inputs: an independently authorized immutable execution manifest; frozen checkpoint hashes;
ten distinct physical controller-evaluation demands per profile; paired SUMO seeds; fixed-time
secondary comparator; exact output/metric/failure contracts; storage budget; and frozen analysis
code. A strong adaptive comparator remains conditional on the documented seven-TLS topology and
traffic-engineering limitations.

Execution requirement: a new preflight must verify the whole manifest and every artifact before
starting SUMO. Existing `evaluate.py` run-level provenance is necessary but not sufficient.

Outputs: complete per-run XML/manifests, physical action traces, invalid/adverse-run register,
paired primary outcomes and analysis-ready policy/demand cluster identifiers.

## Gate 5 — statistical analysis and claims

Inputs: all prespecified cells or an auditable failure register; no seed substitution or selective
rerun. Analyze profile-specific effects and their equal-profile aggregate. Cluster by trained
policy and physical demand while retaining arm pairing and profile strata. Report all five
seed-level effects and sensitivity because five policies provide weak tail resolution.

Outputs: effect estimates, uncertainty, missingness/failure sensitivity and multiplicity-adjusted
secondary outcomes. A result table is not field evidence without empirical calibration.

## Explicitly deferred or prohibited

- No publication training/evaluation while pairing, forecast seeds, registry, calibration and
  execution authority are unresolved.
- No full Tahrir/Taksim run to “test” an unapproved contract.
- No AI-generated traffic-engineering approval, trust root or reviewer identity.
- No use of historical incompatible weights or reported percentages as new evidence.
- No real-world deployment claim from a SUMO-only study.

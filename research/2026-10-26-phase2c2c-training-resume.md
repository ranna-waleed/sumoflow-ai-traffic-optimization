# Phase 2C.2C — interruption-equivalent research-training restoration

Date: 2026-10-26  
Branch: `research/ieee-reproducibility`  
Starting commit: `61bf94e58f625139b71e5c008038491a6fb76905`

## Scope and conclusion

This phase implemented an engineering-only, fail-closed, completed-episode
training snapshot. It did not train either model, run Tahrir/Taksim, alter an
evaluation checkpoint, or grant scientific approval. CPU synthetic probes show
semantic interruption equivalence for the tested four-update interruption plus
four-update continuation with uniform replay and PER. This is not evidence that
future traffic training is deterministic or scientifically authorized.

Both publication approval trust roots remain empty. Selecting
`--research-mode` therefore fails before configuration, agents, or SUMO are
started (`DeepQN/training/train.py:56-65,87-88`). Legacy training and its
best-effort `load_latest()` behavior remain separate and unchanged.

## Mutable-state inventory

| State | Source-verified owner | Snapshot treatment |
|---|---|---|
| Per-TLS online/target networks | `DQNAgent.online_net`, `target_net` | Persisted per agent; strict tensor-shape load into candidates. |
| Adam optimizer tensors/counters | `DQNAgent.optimiser` | Full `state_dict`; optimizer class and complete DQN config hash must match. No scheduler exists in the current implementation. |
| Uniform replay | deque transitions plus isolated `random.Random` | Transitions, capacity, and isolated RNG persisted. |
| PER replay | transition array, priorities, cursor, size, beta and `default_rng` | Complete replay `state_dict` persisted and validated. |
| Exploration/update state | epsilon, total steps, total updates, episode loss accumulator | Persisted per agent. Action selection uses NumPy's global RNG; there is no separate action RNG today. |
| Global RNG | Python, legacy NumPy, Torch CPU, optional all CUDA devices | Actual state is restored after all validation; no reseeding. CUDA-required snapshots reject unavailable or wrong device state. |
| Episode scheduling | completed/next episode, profile cursor, demand cursor, SUMO run index, next SUMO seed | Persisted in `TrainingCursor`; `next_episode` must equal `completed_episode + 1`. |
| Frozen identities | run, run-manifest, dataset, demand registry, split registry, observation, reward, forecast, checkpoint, code/environment | Manifest-bound through `ResumeContracts`; mismatches fail before mutation. |
| Derived state | target-update position | Reproduced by persisted `total_updates`; configuration fixes the update rule/frequency. |
| Not restorable | active SUMO/TraCI process, mid-episode physical state, OS scheduling, nondeterministic external kernels | Explicitly outside the schema. A mid-episode interruption aborts that episode. |

The current seven-agent Tahrir count is not hardcoded: the manifest records and
validates the exact configured ensemble membership, so the same mechanism can
fail closed for another validated topology.

## Boundary and crash semantics

The only valid transaction is: a successful episode ends; all accepted replay
transitions and updates finish; the next episode/profile/demand/SUMO-seed
cursors are frozen; state is captured in staging; every constituent is hashed;
`manifest.json` is written; `COMPLETE.json` is written last; then the staging
directory is atomically renamed to its immutable destination
(`research_resume.py:163-211`). Existing destinations are never overwritten.

An injected write failure removes staging and exposes no snapshot. Missing
markers, modified files, mixed agents/episodes, incompatible contracts, replay,
architecture, optimizer, or required CUDA state are rejected. Load uses
`torch.load(..., weights_only=True)` and builds complete candidate agents first;
the live ensemble is swapped only after all files, identities, tensors, replay,
and RNGs validate (`research_resume.py:214-315`). A caller must supply an
expected manifest SHA-256 obtained through external orchestration. A hash from
the same untrusted source is integrity metadata, not a trust root.

There is no mid-SUMO continuation. A crash before publication means the episode
is discarded and restarted from the last complete snapshot, using the saved
next cursor; partial outputs must not advance the cursor or be accepted.

## Snapshot schemas

- Whole snapshot: `sumoflow-research-training-state-v1`.
- Per-agent artifact: `sumoflow-research-agent-state-v1`.
- Completion marker: `sumoflow-research-training-complete-v1`.
- Historical and research-evaluation checkpoint schemas were not changed.

PyTorch restricted loading reduces executable-pickle exposure but is not a
general untrusted-artifact sandbox. Publication orchestration must obtain the
expected manifest identity independently and control filesystem provenance.

## Deterministic synthetic evidence

`DeepQN/training/resume_probe.py` is a CPU-only engineering probe with one tiny
4-input/2-action agent, replay capacity 32, batch 2, and deterministic Torch.
For each of uniform replay and PER:

1. A fresh process performs `K+M = 4+4` transitions/optimizer opportunities.
2. A second fresh process performs `K=4` and publishes a complete snapshot.
3. That process exits; a third fresh process constructs a new ensemble, loads
   the snapshot, and performs `M=4`.
4. Tests require exact equality of action/epsilon/loss traces and a semantic
   digest covering online/target tensors, optimizer tensors/groups, replay
   transitions, PER priorities/cursors/beta, counters, and replay RNG state.

Both modes passed. Construction-time RNG consumption in the new process was
overridden by restored RNG state. Snapshot byte hashes are not claimed equal:
PyTorch serialization metadata is not a scientific state definition. The test
establishes exact semantic equality only for this bounded CPU fixture.

## Failure evidence

Tests reject incomplete episodes, wrong next-episode numbering, interrupted
publication, overwrite attempts, missing markers, modified constituents,
incorrect external manifest identity, ensemble membership changes, demand
registry changes, observation/reward/forecast contract changes, and
self-consistently rehashed wrong episode/TLS/epsilon payloads. Failure cases
also assert that existing live model parameters remain unchanged.

The replay equivalence probes detect failure to restore uniform/PER sampling
RNG, epsilon, optimizer state, or PER beta/priorities because subsequent exact
traces and final semantic digests would diverge. CUDA was unavailable and no
CUDA equality claim was tested; a CUDA-required snapshot fails closed when its
state cannot be restored.

## Commands and exact results

Focused suite:

```text
python -m unittest DeepQN.tests.test_phase2c2c_training_resume -v
Ran 11 tests in 58.980s
OK
```

Full discovery (this command also ran the repository's pre-existing bounded
SUMO fixture tests; it did not run a full scenario):

```text
python -m unittest discover -s DeepQN/tests -p "test_*.py" -v
Ran 171 tests in 110.134s
OK (skipped=1)
```

The single skip is the pre-existing quarantined interactive
`test_ai_behavior` diagnostic. No Phase 2C.2C test was skipped.

In-memory compilation:

```text
python -c "... compile(Path(p).read_text(...)) ..."
compiled 4 files
```

`git diff --check` exited 0. It emitted only the pre-existing CRLF warning for
the unrelated modified `DeepQN/tests/test_metrics_phase1a.py`. Test-created
tracked `__pycache__` changes were restored. No dependency was installed and no
commit or push occurred.

## Files changed in Phase 2C.2C

- `DeepQN/training/research_resume.py`: new snapshot contracts, atomic publisher,
  restricted loader, candidate validation, and RNG restoration.
- `DeepQN/training/resume_probe.py`: new bounded fresh-process equivalence probe.
- `DeepQN/tests/test_phase2c2c_training_resume.py`: 11 new regression tests.
- `DeepQN/training/train.py`: explicit prohibited research-mode entrypoint;
  legacy default behavior preserved.
- This evidence ledger.

Post-verification SHA-256 identities (source bytes at this stop point):

```text
research_resume.py                    e45b0fc9ec32f63c4880f8d305d0506c55cb5f77b876f3780fbe5722f26f3f8e
resume_probe.py                       956621ede55a8f0fbda04bfba1b80ed1b3320b9d51411fa5c6a200378b9e2d7e
train.py                              718bd42dd39a3c8a1979dbd62f75ea5c51be3c1d7b1f7865d8c580d466bee751
test_phase2c2c_training_resume.py     4d0d91b7ff4a4037e9bec194b78ca2a7da5a034f7ecd28d3ef1ceadf533e7e66
```

Unrelated pre-existing modifications (`.vscode`, backend images,
`test_metrics_phase1a.py`, `DeepQN/forecasting/__init__.py`, and stray status
files) were not edited or staged.

## Determinism qualifications and unresolved blockers

- Exact CPU equivalence is proven only for the small tested architecture,
  versions, and operations. Different BLAS/threading/platform versions and
  future nondeterministic operations require new validation.
- CUDA is untested locally. CUDA device count/version/kernel determinism must be
  frozen and independently tested before any CUDA continuation claim.
- No active SUMO state is persisted; checkpointing remains episode-boundary only.
- There is no scheduler today. Adding one requires a schema/version change and
  mandatory state persistence.
- Environment and code identities are recorded, but an external system must
  create and approve their trust records; self-declared hashes are insufficient.
- Approved Tahrir detector mapping, forecast dataset, demand partition,
  calibration references, compute budget, and human scientific approval remain
  absent. Both publication trust roots are intentionally empty.

## Phase 3 GO/NO-GO

| Gate | Status |
|---|---|
| Complete episode-boundary CPU resume infrastructure | GO for tested engineering scope |
| Uniform/PER fresh-process semantic continuation | GO for bounded synthetic fixture |
| CUDA interruption equivalence | NO-GO / untested |
| Mid-episode continuation | Out of scope and explicitly unsupported |
| Approved forecast/detector/demand/calibration artifacts | NO-GO |
| Authorized publication training | **NO-GO** |
| Phase 3 traffic experiments | **NO-GO** |

Stop point: Phase 2C.2C is complete. Phase 2C.2D and Phase 3 were not started.

## Final research-loop and CUDA-integrity correction

Independent review correctly found that the first implementation supplied a
complete snapshot mechanism but did not connect it to research episode
orchestration. If the empty authority gate had later been populated,
`train(..., research_mode=True)` could have fallen through into the historical
loop, which restarts numbering and calls legacy checkpoint methods. It also let
callers state `cuda_required=False` even when agents were resident on CUDA.

### Corrected production design

`DeepQN/training/research_orchestration.py` now defines a frozen, hashed
`ResearchSchedule`, exact `EpisodePlan`, finalized `EpisodeResult`, and
`run_research_episode_transactions()` (`:12-109`). At each boundary it:

1. validates the current episode/profile/demand/SUMO-run cursors against the
   frozen schedule;
2. gives the injected runner the exact next episode, profile, registered demand
   identity, SUMO seed, and run index;
3. rejects unsuccessful or unfinalized episodes without publishing a snapshot
   or advancing the cursor;
4. constructs the next cursor only after episode finalization;
5. embeds the episode record and schedule hash in the immutable snapshot; and
6. exposes that snapshot as the authoritative episode log. Any tabular log is a
   recoverable/idempotent derivative; a log without its complete snapshot is
   not a committed episode.

Duplicate episode publication encounters the existing non-overwrite check and
fails. A mid-episode process failure invalidates the in-memory process and
restarts from the last complete snapshot; the system does not attempt rollback
inside a damaged process.

`train_research()` is a separate future production pathway
(`DeepQN/training/train.py:73-102`). It checks external authority before reading
configuration, constructing agents, restoring state, or allowing an injected
episode runner to start an environment. It either starts from a complete
explicit cursor or calls `restore_training_snapshot()` with the external
manifest identity and exact schedule identity. It never calls `load_latest()`
or legacy per-agent `save()`. The ordinary `train()` also refuses to fall
through to legacy behavior when `research_mode=True`. Because
`RESEARCH_TRAINING_TRUST_ROOTS` remains empty, both entrypoints fail before any
real environment startup.

### CUDA correction

Snapshot publication now derives CUDA obligation from every live agent device
(`research_resume.py:173-179`); a contradictory caller override is rejected.
The manifest/global state record exact per-agent device strings, CUDA device
count and device names. Restore requires identical live device placement,
CUDA requirement, device count/names, and one valid RNG state per CUDA device.
CPU Torch and CUDA RNG states are validated with candidate `torch.Generator`
objects before the live ensemble is swapped.

This host reported:

```text
cuda_available= False
cuda_device_count= 0
```

Therefore CUDA equivalence remains untested. CPU-only adversarial tests changed
CUDA metadata to require a fabricated GPU and confirmed rejection before model,
replay, Python RNG, NumPy RNG, or Torch RNG mutation. CPU publishers also reject
an explicit `cuda_required=True` override.

### Episode-level fresh-process equivalence

The probe now uses two configured agents and an explicit four-entry schedule:

```text
profiles:       morning, evening, morning, evening
demand IDs:     demand-a, demand-b, demand-a, demand-b
SUMO seeds:     101, 202, 303, 404
K + M:          2 + 2 completed synthetic episodes
steps/episode:  2
```

For uniform replay and PER, the uninterrupted four-episode process exactly
matched two episodes followed by snapshot/process exit/fresh process restore/two
episodes. Assertions cover episode plans, episode-record identities, action,
epsilon and loss sequences, both agents' online/target parameters, optimizer
tensors/groups, replay contents/RNG, PER priorities/cursor/beta, counters,
per-agent loss accumulators, final Python/NumPy/Torch RNG states, and the final
orchestration cursor. Corrupted profile, demand, and SUMO-seed cursors are
explicitly rejected by schedule preflight rather than merely serialized.

### Additional fail-closed checks

- `ResumeContracts.validate()` now requires the supported checkpoint-contract
  version, a Git identity, and a real environment SHA-256.
- Research restoration can require the exact schedule hash embedded in the
  episode record.
- Tests cover an unsuccessful/unfinalized episode, invalid schedule cursors,
  absent authority before even a nonexistent configuration is opened, and CUDA
  metadata failure without ensemble/replay/global-RNG mutation.
- Existing demand/dataset/split/run and observation/reward/forecast identities
  remain exact manifest-bound requirements.

### Final correction verification

```text
python -m unittest DeepQN.tests.test_phase2c2c_training_resume -v
Ran 15 tests in 61.577s
OK

python -c "... compile(...) ..."
compiled 5 files

python -m unittest discover -s DeepQN/tests -p "test_*.py" -v
Ran 175 tests in 110.623s
OK (skipped=1)
```

The only skip remained the pre-existing quarantined interactive diagnostic.
`git diff --check` exited zero and emitted only the pre-existing CRLF warning
for unrelated `DeepQN/tests/test_metrics_phase1a.py`. Test-created tracked cache
files were restored. No scientific training, full scenario, installation,
commit, or push occurred.

Final correction source hashes (superseding earlier hashes above):

```text
research_resume.py                    9c4198ec9814863ffdb0ceb4762f97cfdf7a4323fe3b6c2d42464339a23f0fa7
research_orchestration.py             b9e828d5b02ee8d10492ee61582fca96094eab5bae4e47db335a97db227a25ad
resume_probe.py                       460f9f659a584a571677c0ae23dcbbedb1c9205f250f9a605965f6db83799d88
train.py                              14337e398bee60cc1ce6450f395253c32c64668b5f4d71cbf0963e9118b5975c
test_phase2c2c_training_resume.py     d1deead632a0d40d8850a2b8047a3168f31cf307a1920235d79d513b65451870
```

### Remaining restrictions

CUDA and cross-platform equivalence remain NO-GO; no CUDA device was available.
The synthetic runner establishes orchestration mechanics, not SUMO performance
or full-Tahrir determinism. External approval trust roots, approved detector and
forecast artifacts, calibration, compute budget, and publication authorization
remain absent. Phase 3 remains **NO-GO**.

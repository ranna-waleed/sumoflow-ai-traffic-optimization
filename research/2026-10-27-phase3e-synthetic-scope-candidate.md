# Phase 3.0E synthetic-study scope candidate

Date: 2026-10-27 (Africa/Cairo)  
Status: **candidate only; not an execution authorization**

## Intended claim boundary

The proposed study is an **uncalibrated, Tahrir-derived synthetic simulation study**. It may test
controller behavior on a network derived from the Tahrir map under explicitly generated synthetic
demand. It must not be described as a field-calibrated Tahrir case study, measured Cairo traffic,
real-world deployment, or evidence of real-world benefit. Historical route files are templates;
they are not independent observed demand realizations.

Software-contract tests can establish deterministic generation, identity binding, causal data
access and fail-closed execution. They cannot establish detector placement validity, signal safety,
empirical demand calibration, model performance or publication approval.

## Frozen and unresolved physical contracts

| Area | Source evidence | Candidate treatment | Remaining decision |
|---|---|---|---|
| Network | `simulation/maps/tahrirupdated.net.xml`, SHA-256 `a7f25235f208db932735983653e00b3af659d09d04169ef5807265433c2db708` | Immutable topology input | Field correspondence is not established. |
| Forecast boundaries | `DeepQN/configs/forecast_mappings/tahrir_v1.json`, SHA-256 `275d8494f9a408ddee515b63402bdac27bba041730726231d35f428ac04c0f98` | Structurally checked candidate only | Independent traffic-engineering approval remains absent; publication forecast training stays prohibited. |
| Signal programs | `DeepQN/configs/controller_contracts/tahrir_controller_review_candidate_v1.json`, SHA-256 `2ee35e47b01a9fa5c5866f3e829ccfaa12bdd2a018d39a77031fff73032d47dd` | Source-derived inventory only | Reviewer identity and decisions remain empty. |
| Reward references | `research/2026-10-27-phase3c-reward-calibration-protocol.json`, SHA-256 `0bf45ff5bcb2f265227178b9f3aa425a7948fc1366173079c4f9e3ae1ae51740` | Protocol is frozen; values are absent | Generate training/calibration-only traces after demand freeze. |

## TLS and detector feasibility decisions

| TLS/program class | Forecast E1 candidate | Controller feasibility | Phase 3.0E decision |
|---|---|---|---|
| `315744796`, `2031414903` | Candidate mapping exists; external geometry approval absent | Two competing green phases support the restricted binary next-phase/max-pressure contract | Software preparation only; no scientific run until mapping and program decisions are independently recorded. |
| Other five DQN-controlled TLS | Candidate mapping exists; external geometry approval absent | Single service phase/red-hold topology does not support conventional multi-phase max pressure | DQN/fixed behavior may be studied only under the documented topology; reject a seven-TLS max-pressure claim. |
| `96621100` | Included in source inventory | Native order is all-red -> all-yellow -> all-green -> all-red, while commanded transitions are a different mechanism | Safety/intent cannot be established from source. External review is required; do not silently reorder phases. |
| Two remaining frozen native TLS | Not part of the seven-agent DQN ensemble | Native program only unless a separate contract is approved | Keep outside learned-controller claims and preserve exact program identity. |

The 46-lane E1 inventory passed bounded structural and fixture checks. Those checks do not resolve
real Tahrir lane changes, teleport entry, cyclic re-entry, arbitrary short-edge traversal, or the
traffic-engineering suitability of the chosen boundaries.

## Candidate comparison language

Any future result must identify the frozen network, effective SUMO configuration, demand
realization, seed identities, controller/checkpoint, detector mapping, signal restrictions and
metric contract. Fixed-time, actuated, restricted max-pressure and learned controllers may be
compared only where the effective topology supports each controller and the same physical demand
is used. Random policy remains an engineering diagnostic.

The study remains a candidate until all Gate-0 artifact identities are non-null, independently
reviewed where required, immutable, and accepted through a separately approved execution step.
`publication_approved` and `phase3_authorized` remain false.

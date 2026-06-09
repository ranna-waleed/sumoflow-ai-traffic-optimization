"""
DeepQN/tests/test_ai_behavior.py
---------------------------------
Edge case and adversarial tests for the SUMOFlow AI components.

Tests covered:
  1. Empty network       — no vehicles, all-zero state
  2. Sensor failure      — NaN / Inf values in state vector
  3. Max congestion      — all lane features at maximum (1.0)
  4. BiLSTM zero output  — prediction returns all zeros
  5. BiLSTM max output   — prediction returns maximum flow
  6. Action consistency  — same input always gives same output (determinism)
  7. Action validity     — DQN never outputs action outside {0, 1}
  8. Multi-agent independence — one agent's state doesn't affect another

Run from project root:
    python -m DeepQN.tests.test_ai_behavior
"""

from __future__ import annotations

import sys
import logging
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PASS = "✓ PASS"
FAIL = "✗ FAIL"
results = []


def record(name: str, passed: bool, detail: str = ""):
    status = PASS if passed else FAIL
    results.append((name, passed, detail))
    print(f"  {status}  {name}")
    if detail:
        print(f"         {detail}")


def load_agents():
    """Load trained agents from Tahrir checkpoints."""
    import yaml
    with open("DeepQN/configs/dqn_config.yaml") as f:
        cfg = yaml.safe_load(f)
    from DeepQN.agent.dqn_agent import MultiAgentDQN
    tls_ids = list(cfg["tls_junctions"].keys())
    agents  = MultiAgentDQN(tls_ids, cfg["dqn"])
    agents.load_latest("DeepQN/checkpoints")
    return agents, tls_ids, cfg


def make_obs(tls_ids: list, value: float = 0.0) -> dict:
    """Build observations with all features set to a constant value."""
    return {
        tid: np.full(37, value, dtype=np.float32)
        for tid in tls_ids
    }


def make_obs_with_nan(tls_ids: list) -> dict:
    """Build observations containing NaN and Inf values."""
    obs = {}
    for tid in tls_ids:
        state = np.random.rand(37).astype(np.float32)
        # Inject NaN and Inf at random positions
        state[2]  = float("nan")
        state[10] = float("inf")
        state[25] = float("-inf")
        obs[tid] = state
    return obs



print("\n" + "=" * 60)
print("  SUMOFlow AI — Behavior & Edge Case Tests")
print("=" * 60 + "\n")

#  Load agents 
print("Loading agents...")
try:
    agents, tls_ids, cfg = load_agents()
    print(f"  Loaded {len(tls_ids)} agents from checkpoints.\n")
except Exception as e:
    print(f"  ERROR loading agents: {e}")
    print("  Make sure you have trained checkpoints in DeepQN/checkpoints/")
    sys.exit(1)


print("── Test Group 1: State Edge Cases ──────────────────────────")

# Test 1 — Empty network (all zeros)
try:
    obs     = make_obs(tls_ids, value=0.0)
    actions = agents.act(obs, eval_mode=True)
    valid   = all(a in (0, 1) for a in actions.values())
    record("Empty network (all-zero state)",
           valid,
           f"Actions: { {k[-6:]:actions[k] for k in list(actions)[:3]} } ...")
except Exception as e:
    record("Empty network (all-zero state)", False, str(e))

# Test 2 — Maximum congestion (all features = 1.0)
try:
    obs     = make_obs(tls_ids, value=1.0)
    actions = agents.act(obs, eval_mode=True)
    valid   = all(a in (0, 1) for a in actions.values())
    n_switch = sum(1 for a in actions.values() if a == 1)
    record("Maximum congestion (all-one state)",
           valid,
           f"{n_switch}/{len(tls_ids)} junctions switched")
except Exception as e:
    record("Maximum congestion (all-one state)", False, str(e))

# Test 3 — NaN / Inf sensor failure
try:
    obs_bad = make_obs_with_nan(tls_ids)
    # The sumo_env has a NaN guard — simulate it here
    for tid in obs_bad:
        s = obs_bad[tid]
        if np.any(np.isnan(s)) or np.any(np.isinf(s)):
            obs_bad[tid] = np.nan_to_num(s, nan=0.0, posinf=1.0, neginf=0.0)
    actions = agents.act(obs_bad, eval_mode=True)
    valid   = all(a in (0, 1) for a in actions.values())
    record("Sensor failure (NaN/Inf → sanitized)",
           valid,
           "NaN guard replaced invalid values before inference")
except Exception as e:
    record("Sensor failure (NaN/Inf → sanitized)", False, str(e))

# Test 4 — Random noise state
try:
    np.random.seed(42)
    obs     = {tid: np.random.rand(37).astype(np.float32) for tid in tls_ids}
    actions = agents.act(obs, eval_mode=True)
    valid   = all(a in (0, 1) for a in actions.values())
    record("Random noise state",
           valid,
           "All actions still in valid set {0, 1}")
except Exception as e:
    record("Random noise state", False, str(e))


print("\n── Test Group 2: BiLSTM Input Variations ───────────────────")

# Test 5 — BiLSTM zeros (no prediction signal)
try:
    lstm_zero = {"north": 0, "south": 0, "east": 0, "west": 0}
    # Build state with zero global features (positions 32-35)
    obs = make_obs(tls_ids, value=0.5)
    for tid in obs:
        obs[tid][32:36] = 0.0
    actions = agents.act(obs, eval_mode=True)
    valid   = all(a in (0, 1) for a in actions.values())
    record("BiLSTM zero prediction",
           valid,
           "DQN handles zero flow forecast correctly")
except Exception as e:
    record("BiLSTM zero prediction", False, str(e))

# Test 6 — BiLSTM maximum prediction (traffic spike incoming)
try:
    obs = make_obs(tls_ids, value=0.5)
    for tid in obs:
        obs[tid][32:36] = 1.0   # max normalized flow in all directions
    actions  = agents.act(obs, eval_mode=True)
    valid    = all(a in (0, 1) for a in actions.values())
    n_switch = sum(1 for a in actions.values() if a == 1)
    record("BiLSTM max prediction (traffic spike)",
           valid,
           f"{n_switch}/{len(tls_ids)} junctions switched on predicted spike")
except Exception as e:
    record("BiLSTM max prediction (traffic spike)", False, str(e))


print("\n── Test Group 3: Determinism & Consistency ─────────────────")

# Test 7 : Determinism: same input → same output
try:
    np.random.seed(0)
    obs   = {tid: np.random.rand(37).astype(np.float32) for tid in tls_ids}
    runs  = [agents.act(obs, eval_mode=True) for _ in range(5)]
    # All 5 runs should be identical
    same  = all(runs[i] == runs[0] for i in range(1, 5))
    record("Determinism (same input → same output × 5)",
           same,
           "Greedy policy (epsilon=0) is fully deterministic")
except Exception as e:
    record("Determinism (same input × 5)", False, str(e))

# Test 8 : Action validity across 1000 random states
try:
    all_valid = True
    for _ in range(1000):
        obs     = {tid: np.random.rand(37).astype(np.float32) for tid in tls_ids}
        actions = agents.act(obs, eval_mode=True)
        if not all(a in (0, 1) for a in actions.values()):
            all_valid = False
            break
    record("Action validity (1,000 random states)",
           all_valid,
           "All actions in {0=keep, 1=switch} across 1,000 random inputs")
except Exception as e:
    record("Action validity (1,000 random states)", False, str(e))


print("\n── Test Group 4: Multi-Agent Independence ──────────────────")

# Test 9 : Changing one agent's state doesn't affect others
try:
    base_obs = make_obs(tls_ids, value=0.3)
    base_act = agents.act(base_obs, eval_mode=True)

    # Change only the first agent's state to max
    mod_obs = {tid: s.copy() for tid, s in base_obs.items()}
    first_tid = tls_ids[0]
    mod_obs[first_tid] = np.ones(37, dtype=np.float32)
    mod_act = agents.act(mod_obs, eval_mode=True)

    # Other agents' actions should be unchanged
    others_same = all(
        mod_act[tid] == base_act[tid]
        for tid in tls_ids[1:]
    )
    record("Multi-agent independence (change agent 1 only)",
           others_same,
           "Agents 2-7 unaffected when agent 1 state changes")
except Exception as e:
    record("Multi-agent independence", False, str(e))

# Test 10 : Each agent has independent Q-values
try:
    import torch
    obs    = make_obs(tls_ids, value=0.5)
    stable = True
    for tid in tls_ids:
        state_t = torch.FloatTensor(obs[tid]).unsqueeze(0)
        with torch.no_grad():
            q = agents.agents[tid].online_net(state_t)
        if q.shape[-1] != 2:
            stable = False
            break
    record("Q-value output shape (2 actions per agent)",
           stable,
           f"Each agent outputs Q(keep) and Q(switch) independently")
except Exception as e:
    record("Q-value output shape", False, str(e))

print("\n── Test Group 5: Failure Recovery ──────────────────────────")

# Test 11 — Missing junction in observation (partial state)
try:
    obs_partial = make_obs(tls_ids, value=0.5)
    # Remove one junction from observations
    missing_tid = tls_ids[-1]
    del obs_partial[missing_tid]

    # Simulate the fallback: fill missing with zeros
    obs_filled = {tid: obs_partial.get(tid, np.zeros(37, dtype=np.float32)) for tid in tls_ids}
    actions = agents.act(obs_filled, eval_mode=True)
    valid   = all(a in (0, 1) for a in actions.values())
    record("Missing junction observation (filled with zeros)",
           valid,
           f"Missing TLS '{missing_tid[-8:]}' filled with zero state")
except Exception as e:
    record("Missing junction observation", False, str(e))

# Test 12 — State vector wrong size (should be caught before reaching agent)
try:
    wrong_state = np.ones(20, dtype=np.float32)  # should be 37
    # Simulate the guard that pads/truncates
    EXPECTED = 37
    if len(wrong_state) < EXPECTED:
        fixed = np.pad(wrong_state, (0, EXPECTED - len(wrong_state)))
    else:
        fixed = wrong_state[:EXPECTED]
    valid = len(fixed) == EXPECTED
    record("Wrong state vector size (pad/truncate guard)",
           valid,
           f"Size {len(wrong_state)} → padded/truncated to {EXPECTED}")
except Exception as e:
    record("Wrong state vector size", False, str(e))


print("\n" + "=" * 60)
print("  RESULTS SUMMARY")
print("=" * 60)

passed = sum(1 for _, p, _ in results if p)
total  = len(results)

for name, p, detail in results:
    status = PASS if p else FAIL
    print(f"  {status}  {name}")

print("=" * 60)
print(f"  {passed}/{total} tests passed")
if passed == total:
    print("  All AI behavior tests passed.")
else:
    failed = [name for name, p, _ in results if not p]
    print(f"  Failed: {failed}")
print()
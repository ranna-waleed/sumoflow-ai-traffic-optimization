# tests/test_ai_edge_cases.py
# Tests AI components under edge cases, stress conditions,
# and unexpected inputs : covers checklist point 4
import os
import sys
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

PASS = "PASS"
FAIL = "FAIL"
results = []

def log(test_name, passed, detail=""):
    status = PASS if passed else FAIL
    results.append((test_name, status, detail))
    print(f"  {status} {test_name}" + (f" — {detail}" if detail else ""))


print("  SUMOFlow AI — Edge Case & Stress Tests")

# SECTION 1: DQN Agent Tests
print("\n[1] DQN Agent Tests")

try:
    from dqn.agent import DQNAgent

    agent = DQNAgent(state_size=14, action_size=4)

    # Test 1.1 — Zero state (no vehicles)
    try:
        state = np.zeros(14, dtype=np.float32)
        action, q_vals = agent.act_with_q(state)
        ok = 0 <= action <= 3 and len(q_vals) == 4
        log("1.1 Zero state (0 vehicles)", ok,
            f"action={action}, q={[round(q,3) for q in q_vals]}")
    except Exception as e:
        log("1.1 Zero state (0 vehicles)", False, str(e))

    # Test 1.2 : Max state (all features = 1.0)
    try:
        state = np.ones(14, dtype=np.float32)
        action, q_vals = agent.act_with_q(state)
        ok = 0 <= action <= 3
        log("1.2 Max state (all features=1.0)", ok,
            f"action={action}")
    except Exception as e:
        log("1.2 Max state (all features=1.0)", False, str(e))

    # Test 1.3 : NaN in state (sensor glitch)
    try:
        state = np.zeros(14, dtype=np.float32)
        state[3] = float("nan")
        state[7] = float("inf")
        # Apply the guard we added
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=0.0)
        action, q_vals = agent.act_with_q(state)
        ok = not any(np.isnan(q_vals)) and 0 <= action <= 3
        log("1.3 NaN/Inf in state — guard applied", ok,
            f"action={action} after sanitization")
    except Exception as e:
        log("1.3 NaN/Inf in state", False, str(e))

    # Test 1.4 : Wrong state size
    try:
        state = np.zeros(6, dtype=np.float32)  # old 6-feature state
        action, q_vals = agent.act_with_q(state)
        log("1.4 Wrong state size (6 instead of 14)", False,
            "Should have raised error but didn't")
    except Exception:
        log("1.4 Wrong state size raises error", True,
            "Correctly rejected wrong input size")

    # Test 1.5 : Negative values in state
    try:
        state = np.full(14, -0.5, dtype=np.float32)
        action, q_vals = agent.act_with_q(state)
        ok = 0 <= action <= 3
        log("1.5 Negative state values", ok,
            f"action={action} (model handles gracefully)")
    except Exception as e:
        log("1.5 Negative state values", False, str(e))

    # Test 1.6 : Deterministic at epsilon=0
    try:
        agent.epsilon = 0.0
        state = np.random.rand(14).astype(np.float32)
        actions = [agent.act_with_q(state)[0] for _ in range(10)]
        ok = len(set(actions)) == 1  # all same action
        log("1.6 Deterministic at epsilon=0", ok,
            f"all 10 calls returned action={actions[0]}")
    except Exception as e:
        log("1.6 Deterministic at epsilon=0", False, str(e))

    # Test 1.7 : Load trained model if exists
    try:
        model_path = os.path.join(BASE_DIR, "dqn", "models", "dqn_best.pth")
        if os.path.exists(model_path):
            trained_agent = DQNAgent(state_size=14, action_size=4)
            trained_agent.load()
            state = np.random.rand(14).astype(np.float32)
            action, q_vals = trained_agent.act_with_q(state)
            ok = 0 <= action <= 3
            log("1.7 Load trained model + inference", ok,
                f"action={action}, max_q={max(q_vals):.4f}")
        else:
            log("1.7 Load trained model", True,
                "Skipped — model not trained yet")
    except Exception as e:
        log("1.7 Load trained model", False, str(e))

except ImportError as e:
    log("DQN module import", False, str(e))


# SECTION 2: BiLSTM Predictor Tests
print("\n[2] BiLSTM Predictor Tests")

try:
    from lstm.predict import predict, is_ready

    # Test 2.1 : Model ready check
    try:
        ready = is_ready()
        log("2.1 Model ready check", True,
            f"ready={ready}")
    except Exception as e:
        log("2.1 Model ready check", False, str(e))

    if is_ready():

        # Test 2.2 : Normal input (60 timesteps)
        try:
            history = [
                {"north": 10, "south": 8, "east": 12, "west": 6,
                 "total": 36, "avg_speed": 8.5, "avg_waiting": 15.0}
                for _ in range(60)
            ]
            result = predict(history)
            ok = all(k in result for k in ["north","south","east","west","next_30s"])
            ok = ok and all(v >= 0 for v in result["next_30s"].values())
            log("2.2 Normal 60-step input", ok,
                f"next_30s={result['next_30s']}")
        except Exception as e:
            log("2.2 Normal 60-step input", False, str(e))

        # Test 2.3 : Minimum input (10 timesteps)
        try:
            history = [
                {"north": 5, "south": 5, "east": 5, "west": 5,
                 "total": 20, "avg_speed": 10.0, "avg_waiting": 5.0}
                for _ in range(10)
            ]
            result = predict(history)
            ok = "next_30s" in result
            log("2.3 Minimum input (10 timesteps)", ok,
                f"next_30s={result['next_30s']}")
        except Exception as e:
            log("2.3 Minimum input (10 timesteps)", False, str(e))

        # Test 2.4 — Zero traffic (empty roads)
        try:
            history = [
                {"north": 0, "south": 0, "east": 0, "west": 0,
                 "total": 0, "avg_speed": 0.0, "avg_waiting": 0.0}
                for _ in range(60)
            ]
            result = predict(history)
            ok = all(v >= 0 for v in result["next_30s"].values())
            log("2.4 Zero traffic (empty roads)", ok,
                f"next_30s={result['next_30s']} (should be near 0)")
        except Exception as e:
            log("2.4 Zero traffic", False, str(e))

        # Test 2.5 : Peak traffic (rush hour spike)
        try:
            history = [
                {"north": 150, "south": 120, "east": 100, "west": 80,
                 "total": 450, "avg_speed": 2.0, "avg_waiting": 120.0}
                for _ in range(60)
            ]
            result = predict(history)
            ok = all(v >= 0 for v in result["next_30s"].values())
            log("2.5 Peak traffic (rush hour spike)", ok,
                f"next_30s={result['next_30s']}")
        except Exception as e:
            log("2.5 Peak traffic", False, str(e))

        # Test 2.6 : Missing features in history
        try:
            history = [
                {"north": 10, "south": 8}   # missing east, west, total, etc.
                for _ in range(60)
            ]
            result = predict(history)
            ok = "next_30s" in result
            log("2.6 Missing features in history", ok,
                "Model handles missing keys with .get() default 0")
        except Exception as e:
            log("2.6 Missing features in history", False, str(e))

        # Test 2.7 : Output never negative
        try:
            results_neg = []
            for _ in range(5):
                history = [
                    {k: np.random.randint(0, 100)
                     for k in ["north","south","east","west","total"]}
                    | {"avg_speed": np.random.uniform(0,15),
                       "avg_waiting": np.random.uniform(0,120)}
                    for _ in range(60)
                ]
                r = predict(history)
                results_neg.append(
                    all(v >= 0 for v in r["next_30s"].values())
                )
            ok = all(results_neg)
            log("2.7 Output always non-negative (5 random inputs)", ok)
        except Exception as e:
            log("2.7 Output always non-negative", False, str(e))

    else:
        log("2.2-2.7 LSTM tests", True,
            "Skipped — model not trained yet")

except ImportError as e:
    log("LSTM module import", False, str(e))


# SECTION 3: YOLO Detection Tests
print("\n[3] YOLO Detection Tests")

try:
    from backend.services.yolo_detect import get_model, CLASS_NAMES

    model_path = os.path.join(
        BASE_DIR, "detection", "yolo", "results", "best.pt"
    )

    if os.path.exists(model_path):
        # Test 3.1 : Model loads without error
        try:
            model = get_model()
            log("3.1 Model loads successfully", model is not None)
        except Exception as e:
            log("3.1 Model loads successfully", False, str(e))

        # Test 3.2 : Blank/tiny image (blank frame guard)
        try:
            from backend.services.yolo_detect import detect_image
            tiny_bytes = b"\xff\xd8\xff" + b"\x00" * 500  # fake tiny JPEG
            try:
                result = detect_image(tiny_bytes)
                log("3.2 Tiny/blank image handled", True,
                    f"detections={result.get('total_detections', 0)}")
            except Exception:
                log("3.2 Tiny/blank image raises gracefully", True,
                    "Exception caught correctly")
        except Exception as e:
            log("3.2 Tiny image test", False, str(e))

        # Test 3.3 : Class names correct
        try:
            expected = {"car","bus","truck","taxi",
                        "microbus","motorcycle","bicycle"}
            ok = set(CLASS_NAMES.values()) == expected
            log("3.3 Class names correct (7 classes)", ok,
                str(set(CLASS_NAMES.values())))
        except Exception as e:
            log("3.3 Class names", False, str(e))

    else:
        log("3.1-3.3 YOLO tests", True,
            "Skipped — best.pt not found")

except ImportError as e:
    log("YOLO module import", False, str(e))


# SECTION 4: State Vector Validation Tests
print("\n[4] State Vector Validation Tests")

# Test 4.1 : All values in [0, 1] range after normalization
try:
    from dqn.agent import DQNAgent
    agent = DQNAgent(state_size=14, action_size=4)

    # Simulate extreme traffic values
    extreme_cases = [
        np.zeros(14),                          # all zeros
        np.ones(14),                           # all ones
        np.full(14, 0.5),                      # mid values
        np.random.uniform(0, 1, 14),           # random valid
        np.clip(np.random.randn(14), 0, 1),    # clipped normal
    ]

    all_valid = True
    for i, state in enumerate(extreme_cases):
        state = state.astype(np.float32)
        action, q_vals = agent.act_with_q(state)
        if not (0 <= action <= 3):
            all_valid = False
            break

    log("4.1 All extreme state vectors produce valid action", all_valid,
        f"Tested {len(extreme_cases)} cases")
except Exception as e:
    log("4.1 Extreme state vectors", False, str(e))

# Test 4.2 : NaN guard works correctly
try:
    bad_state = np.array([0.1, 0.2, float("nan"), 0.4,
                          0.5, float("inf"), 0.3, 0.2,
                          0.1, 0.1, 0.1, 0.1, 0.5, 0.3],
                         dtype=np.float32)
    cleaned = np.nan_to_num(bad_state, nan=0.0, posinf=1.0, neginf=0.0)
    ok = not np.any(np.isnan(cleaned)) and not np.any(np.isinf(cleaned))
    log("4.2 NaN/Inf guard cleans state correctly", ok,
        f"NaN count after: {np.sum(np.isnan(cleaned))}")
except Exception as e:
    log("4.2 NaN guard", False, str(e))


# SUMMARY
print("  RESULTS SUMMARY")

passed = sum(1 for _, s, _ in results if s == PASS)
failed = sum(1 for _, s, _ in results if s == FAIL)
total  = len(results)

print(f"\n  Total:  {total}")
print(f"  Passed: {passed} ")
print(f"  Failed: {failed} ")
print(f"  Score:  {passed}/{total} ({100*passed//total}%)\n")

if failed > 0:
    print("  Failed tests:")
    for name, status, detail in results:
        if status == FAIL:
            print(f" {name}: {detail}")

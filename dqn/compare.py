# dqn/compare.py
# Compare fixed timing vs DQN results
# Run AFTER run_baseline.py and run_dqn.py: python dqn/compare.py

import os, sys, json
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

BASELINE_PATH = os.path.join(RESULTS_DIR, "baseline_results.json")
DQN_PATH      = os.path.join(RESULTS_DIR, "dqn_results.json")
OUTPUT_PATH   = os.path.join(RESULTS_DIR, "comparison_results.json")


def main():
    print("  SUMOFlow AI — Before vs After Comparison")

    if not os.path.exists(BASELINE_PATH):
        print("  ERROR: Run python dqn/run_baseline.py first")
        return
    if not os.path.exists(DQN_PATH):
        print("  ERROR: Run python dqn/run_dqn.py first")
        return

    with open(BASELINE_PATH) as f:
        baseline = json.load(f)
    with open(DQN_PATH) as f:
        dqn = json.load(f)

    profiles = list(dqn["profiles"].keys())

    comparison = {}
    for profile in profiles:
        if profile not in baseline["profiles"]:
            continue

        b = baseline["profiles"][profile]
        d = dqn["profiles"][profile]

        wait_improvement = (b["avg_wait_s"] - d["avg_wait_s"]) / max(b["avg_wait_s"], 0.001) * 100
        co2_improvement  = (b["avg_co2_mg"] - d["avg_co2_mg"]) / max(b["avg_co2_mg"], 0.001) * 100

        comparison[profile] = {
            "fixed_wait_s":      b["avg_wait_s"],
            "dqn_wait_s":        d["avg_wait_s"],
            "wait_improvement":  round(wait_improvement, 2),
            "fixed_co2_mg":      b["avg_co2_mg"],
            "dqn_co2_mg":        d["avg_co2_mg"],
            "co2_improvement":   round(co2_improvement, 2),
        }

        print(f"\n  {profile}:")
        print(f"    Wait:  {b['avg_wait_s']:.3f}s → {d['avg_wait_s']:.3f}s  ({wait_improvement:+.1f}%)")
        print(f"    CO2:   {b['avg_co2_mg']/1000:.1f}k → {d['avg_co2_mg']/1000:.1f}k mg  ({co2_improvement:+.1f}%)")

    # Overall
    b_overall = baseline["overall"]
    d_overall = dqn["overall"]

    overall_wait_imp = (b_overall["avg_wait_s"] - d_overall["avg_wait_s"]) / max(b_overall["avg_wait_s"], 0.001) * 100
    overall_co2_imp  = (b_overall["avg_co2_mg"] - d_overall["avg_co2_mg"]) / max(b_overall["avg_co2_mg"], 0.001) * 100

    result = {
        "profiles":           comparison,
        "overall": {
            "fixed_wait_s":      b_overall["avg_wait_s"],
            "dqn_wait_s":        d_overall["avg_wait_s"],
            "wait_improvement":  round(overall_wait_imp, 2),
            "fixed_co2_mg":      b_overall["avg_co2_mg"],
            "dqn_co2_mg":        d_overall["avg_co2_mg"],
            "co2_improvement":   round(overall_co2_imp, 2),
        },
        # Fields for BeforeAfter.jsx
        "avg_wait_fixed":    b_overall["avg_wait_s"],
        "avg_wait_dqn":      d_overall["avg_wait_s"],
        "improvement_pct":   round(overall_wait_imp, 2),
        "co2_improvement_pct": round(overall_co2_imp, 2),
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  {'='*40}")
    print(f"  OVERALL RESULTS:")
    print(f"    Wait time:  {b_overall['avg_wait_s']:.3f}s → {d_overall['avg_wait_s']:.3f}s  ({overall_wait_imp:+.1f}%)")
    print(f"    CO2:  {b_overall['avg_co2_mg']/1000:.1f}k → {d_overall['avg_co2_mg']/1000:.1f}k mg  ({overall_co2_imp:+.1f}%)")
    print(f"  Saved → {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
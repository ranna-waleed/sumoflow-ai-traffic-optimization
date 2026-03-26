# dqn/run_dqn.py
# Run SUMO with trained DQN controlling signals — get AFTER metrics
# Run AFTER training and baseline: python dqn/run_dqn.py

import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dqn.environment import TahrirEnv
from dqn.agent       import DQNAgent

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

PROFILES = ["morning_rush", "evening_rush", "midday", "night"]


def run_dqn_profile(agent, profile: str) -> dict:
    """Run one simulation with trained DQN controlling signals."""
    print(f"  Running DQN control: {profile}...")

    env   = TahrirEnv(profile=profile, gui=False, port=8815)
    state = env.reset()
    done  = False

    while not done:
        action             = agent.act(state)
        state, _, done, _  = env.step(action)

    summary = env.get_episode_summary()
    env.close()

    return {
        "profile":      profile,
        "avg_wait_s":   summary["avg_wait_s"],
        "avg_co2_mg":   summary["avg_co2_mg"],
        "total_co2_mg": summary["total_co2_mg"],
    }


def main():
    print("  SUMOFlow AI — DQN Controlled Simulation")
    print("  Recording AFTER metrics (with DQN)")

    # Load trained agent
    agent         = DQNAgent(state_size=6, action_size=4)
    agent.load()
    agent.epsilon = 0.0   # pure exploitation

    results = {}
    for profile in PROFILES:
        results[profile] = run_dqn_profile(agent, profile)
        r = results[profile]
        print(f"  {profile:<15}: wait={r['avg_wait_s']:.3f}s | "
              f"CO2={r['avg_co2_mg']/1000:.1f}k mg")

    all_waits = [r["avg_wait_s"] for r in results.values()]
    all_co2   = [r["avg_co2_mg"] for r in results.values()]

    summary = {
        "type":     "dqn_controlled",
        "profiles": results,
        "overall": {
            "avg_wait_s": float(np.mean(all_waits)),
            "avg_co2_mg": float(np.mean(all_co2)),
        }
    }

    out_path = os.path.join(RESULTS_DIR, "dqn_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Overall avg wait:  {summary['overall']['avg_wait_s']:.3f}s")
    print(f"  Overall avg CO2:   {summary['overall']['avg_co2_mg']/1000:.1f}k mg")
    print(f"  Saved → {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
# dqn/visualize.py
# Watch the trained DQN control traffic lights in SUMO-GUI
# Run AFTER training: python dqn/visualize.py

import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dqn.environment import TahrirEnv
from dqn.agent       import DQNAgent

PROFILE = "morning_rush"   # change to see different profiles


def visualize():
    print("=" * 60)
    print("  SUMOFlow AI — DQN Visualization")
    print(f"  Profile: {PROFILE}")
    print("  Watching DQN control traffic lights in real time")
    print("=" * 60)

    # Run with GUI and verbose=True to see decisions
    env = TahrirEnv(profile=PROFILE, gui=True, port=8815, verbose=True)

    agent = DQNAgent(state_size=6, action_size=4)
    agent.load()        # load trained dqn_best.pth
    agent.epsilon = 0.0 # pure exploitation — no random actions

    print(f"[DQN] Loaded trained model | ε=0.0 (pure exploitation)")
    print(f"[DQN] Starting {PROFILE} simulation with SUMO-GUI")
    print()

    state       = env.reset()
    done        = False
    step_count  = 0
    total_wait  = 0.0
    total_co2   = 0.0
    action_log  = []

    while not done:
        action                         = agent.act(state)
        state, reward, done, info      = env.step(action)

        step_count += 1
        total_wait += info["waiting"]
        total_co2  += info["total_co2_mg"]

        action_log.append({
            "step":        info["step"],
            "action":      action,
            "action_name": info["action_name"],
            "wait_s":      round(info["waiting"], 3),
            "co2_mg":      round(info["total_co2_mg"], 1),
            "reward":      round(reward, 4),
        })

    env.close()

    #  Summary 
    print()
    print("=" * 60)
    print("  Visualization Complete!")
    print(f"  Total steps:     {step_count}")
    print(f"  Avg wait time:   {total_wait / max(step_count, 1):.3f}s")
    print(f"  Avg CO2:         {total_co2  / max(step_count, 1) / 1000:.1f}k mg/step")
    print()

    # Action distribution
    from collections import Counter
    action_counts = Counter(log["action"] for log in action_log)
    print("  DQN Action Distribution:")
    action_names = ["All Green", "N-S Green", "E-W Green", "Yellow"]
    for a, name in enumerate(action_names):
        count = action_counts.get(a, 0)
        pct   = count / max(step_count, 1) * 100
        bar   = "█" * int(pct / 5)
        print(f"    Action {a} ({name:<12}): {count:3d} times ({pct:5.1f}%) {bar}")

    # Save log
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results", "visualization_log.json")
    with open(log_path, "w") as f:
        json.dump({
            "profile":      PROFILE,
            "total_steps":  step_count,
            "avg_wait_s":   total_wait / max(step_count, 1),
            "avg_co2_mg":   total_co2  / max(step_count, 1),
            "action_log":   action_log,
            "action_counts": dict(action_counts),
        }, f, indent=2)

    print(f"\n  Decision log saved → dqn/results/visualization_log.json")
    print("=" * 60)


if __name__ == "__main__":
    visualize()
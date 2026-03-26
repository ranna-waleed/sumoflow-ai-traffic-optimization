# dqn/train_dqn.py
import os, sys, json
import numpy as np
import mlflow

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dqn.environment import TahrirEnv
from dqn.agent       import DQNAgent

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MODELS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

EPISODES   = 50
PROFILES   = ["morning_rush", "evening_rush", "midday", "night"]
SAVE_EVERY = 10


def train():
    print("=" * 60)
    print("  SUMOFlow AI — DQN (Smart Real Phases)")
    print("  State: 8 features | Actions: 4 real SUMO phases")
    print(f"  Episodes: {EPISODES} | Profiles: {len(PROFILES)}")
    print("=" * 60)

    env = TahrirEnv(profile=PROFILES[0], gui=False, port=8814)

    agent = DQNAgent(
        state_size    = env.state_size,   # 8
        action_size   = env.action_size,  # 4
        hidden        = 128,
        lr            = 1e-3,
        gamma         = 0.95,
        epsilon       = 1.0,
        epsilon_min   = 0.01,
        epsilon_decay = 0.9995,
        batch_size    = 32,
        memory_size   = 10_000,
        target_update = 5,
    )

    episode_rewards = []
    episode_waits   = []
    episode_co2     = []
    best_avg_reward = float("-inf")

    for episode in range(1, EPISODES + 1):
        profile = PROFILES[(episode - 1) % len(PROFILES)]
        state   = env.reset(profile=profile)

        total_reward = 0.0
        steps        = 0
        done         = False

        while not done:
            action                         = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state        = next_state
            total_reward += reward
            steps        += 1

        if episode % agent.target_update == 0:
            agent.update_target()

        agent.episode = episode
        summary = env.get_episode_summary()
        episode_rewards.append(total_reward)
        episode_waits.append(summary["avg_wait_s"])
        episode_co2.append(summary["avg_co2_mg"])

        avg_r10 = np.mean(episode_rewards[-10:])

        print(f"  Episode {episode:3d}/{EPISODES} | "
              f"profile={profile:<12} | "
              f"reward={total_reward:7.2f} | "
              f"wait={summary['avg_wait_s']:.3f}s | "
              f"CO2={summary['avg_co2_mg']/1000:.1f}k mg | "
              f"ε={agent.epsilon:.3f}")

        if avg_r10 > best_avg_reward:
            best_avg_reward = avg_r10
            agent.save(os.path.join(MODELS_DIR, "dqn_best.pth"))

        if episode % SAVE_EVERY == 0:
            agent.save(os.path.join(MODELS_DIR, f"dqn_ep{episode}.pth"))

    env.close()

    first10_wait = float(np.mean(episode_waits[:10]))
    last10_wait  = float(np.mean(episode_waits[-10:]))
    first10_co2  = float(np.mean(episode_co2[:10]))
    last10_co2   = float(np.mean(episode_co2[-10:]))

    results = {
        "episodes":            EPISODES,
        "profiles":            PROFILES,
        "final_epsilon":       agent.epsilon,
        "best_avg_reward":     best_avg_reward,
        "episode_rewards":     [float(r) for r in episode_rewards],
        "episode_waits":       [float(w) for w in episode_waits],
        "episode_co2":         [float(c) for c in episode_co2],
        "avg_wait_first10":    first10_wait,
        "avg_wait_last10":     last10_wait,
        "improvement_pct":     float((first10_wait - last10_wait) / max(first10_wait, 0.001) * 100),
        "avg_co2_first10":     first10_co2,
        "avg_co2_last10":      last10_co2,
        "co2_improvement_pct": float((first10_co2 - last10_co2) / max(first10_co2, 0.001) * 100),
        "avg_wait_fixed":      first10_wait,
        "avg_wait_dqn":        last10_wait,
    }

    with open(os.path.join(RESULTS_DIR, "training_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ── MLflow tracking ──────────────────────────────────────
    mlflow.set_experiment("SUMOFlow-DQN")
    with mlflow.start_run(run_name=f"ep{EPISODES}_decay{agent.epsilon_decay}"):
        # Log parameters
        mlflow.log_param("episodes",       EPISODES)
        mlflow.log_param("profiles",       str(PROFILES))
        mlflow.log_param("epsilon_decay",  agent.epsilon_decay)
        mlflow.log_param("batch_size",     agent.batch_size)
        mlflow.log_param("gamma",          agent.gamma)
        mlflow.log_param("reward",         "wait_only")
        mlflow.log_param("state_size",     env.state_size)

        # Log results
        mlflow.log_metric("wait_improvement_pct",  results["improvement_pct"])
        mlflow.log_metric("co2_improvement_pct",   results["co2_improvement_pct"])
        mlflow.log_metric("avg_wait_first10",      results["avg_wait_first10"])
        mlflow.log_metric("avg_wait_last10",       results["avg_wait_last10"])
        mlflow.log_metric("avg_co2_first10",       results["avg_co2_first10"])
        mlflow.log_metric("avg_co2_last10",        results["avg_co2_last10"])
        mlflow.log_metric("best_avg_reward",       results["best_avg_reward"])

        # Log files
        mlflow.log_artifact(os.path.join(RESULTS_DIR, "training_results.json"))
        mlflow.log_artifact(os.path.join(MODELS_DIR,  "dqn_best.pth"))

        print(f"  MLflow run logged")

if __name__ == "__main__":
    train()
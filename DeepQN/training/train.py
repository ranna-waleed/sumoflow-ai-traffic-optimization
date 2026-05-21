"""
dqn/training/train.py
---------------------
Main training entry-point for SUMOFlow AI DQN.

Usage
-----
    python -m dqn.training.train
    python -m dqn.training.train --config dqn/configs/dqn_config.yaml
    python -m dqn.training.train --profile morning_rush --episodes 50
    python -m dqn.training.train --resume dqn/checkpoints   # continue training

Algorithm
---------
For each episode:
  1. Pick a traffic profile (sequential or random rotation).
  2. Reset the SUMO environment.
  3. Loop decision steps until the simulation ends:
       a. Every agent selects an action (epsilon-greedy).
       b. Actions applied to SUMO, simulation advanced.
       c. Reward and next-state collected.
       d. Transitions stored; agents updated.
  4. Log metrics; checkpoint every N episodes.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from DeepQN.agent.dqn_agent import MultiAgentDQN
from DeepQN.env.sumo_env import SumoEnv
from DeepQN.training.callbacks import TrainingLogger, EpisodeCallback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


#  Training loop 

def train(
    config_path: str = "DeepQN/configs/dqn_config.yaml",
    override_profile: Optional[str] = None,
    num_episodes:     Optional[int]  = None,
    resume_dir:       Optional[str]  = None,
    port:             int             = 8813,
):
    """
    Full training procedure.

    Parameters
    ----------
    config_path      : path to dqn_config.yaml
    override_profile : force a single profile (default: rotate through all 4)
    num_episodes     : override yaml setting
    resume_dir       : load latest checkpoints from this directory
    port             : TraCI base port
    """
    #  Load config 
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    train_cfg    = cfg["training"]
    dqn_cfg      = cfg["dqn"]
    tls_ids      = list(cfg["tls_junctions"].keys())

    n_episodes   = num_episodes or train_cfg.get("num_episodes", 200)
    ckpt_dir     = train_cfg.get("checkpoint_dir", "DeepQN/checkpoints")
    ckpt_freq    = train_cfg.get("checkpoint_freq", 10)
    log_dir      = train_cfg.get("log_dir",  "DeepQN/logs")
    seed         = train_cfg.get("seed", 42)
    profile_list = [override_profile] if override_profile else (
        train_cfg.get("profiles_per_episode", list(cfg["profiles"].keys()))
    )
    rotation = train_cfg.get("profile_rotation", "sequential")

    #  Reproducibility 
    random.seed(seed)
    np.random.seed(seed)

    #  Agents 
    multi_agent = MultiAgentDQN(tls_ids, dqn_cfg)
    if resume_dir:
        multi_agent.load_latest(resume_dir)

    #  Logger 
    tlog = TrainingLogger(log_dir)

    logger.info(
        "Training started: %d episodes, %d agents, profiles=%s",
        n_episodes, len(tls_ids), profile_list,
    )

    #  Episode loop 
    env: Optional[SumoEnv] = None

    for episode in range(1, n_episodes + 1):
        # Pick profile
        if rotation == "random":
            profile = random.choice(profile_list)
        else:
            profile = profile_list[(episode - 1) % len(profile_list)]

        ep_start = time.time()

        # Create fresh env for each episode (new SUMO process)
        env = SumoEnv(cfg, profile=profile, port=port)
        obs = env.reset()

        ep_reward_sum  = {tid: 0.0 for tid in tls_ids}
        ep_steps       = 0
        done           = False

        #  Decision loop 
        while not done:
            actions  = multi_agent.act(obs, eval_mode=False)
            next_obs, rewards, done, info = env.step(actions)

            # Store experience
            multi_agent.push(obs, actions, rewards, next_obs, done)

            # Update all agents
            multi_agent.update()

            # Accumulate
            for tid in tls_ids:
                ep_reward_sum[tid] += rewards.get(tid, 0.0)
            ep_steps += 1
            obs = next_obs

        env.close()
        env = None

        #  Episode stats 
        agent_stats = multi_agent.episode_stats()
        ep_time     = time.time() - ep_start
        mean_reward = float(np.mean(list(ep_reward_sum.values())))

        tlog.log_episode(
            episode     = episode,
            profile     = profile,
            mean_reward = mean_reward,
            ep_steps    = ep_steps,
            ep_time_s   = ep_time,
            agent_stats = agent_stats,
            info        = info,
        )

        logger.info(
            "Ep %4d  profile=%-14s  reward=%.2f  steps=%d  eps=%.3f  t=%.1fs",
            episode,
            profile,
            mean_reward,
            ep_steps,
            list(multi_agent.agents.values())[0].epsilon,
            ep_time,
        )

        #  Checkpoint 
        if episode % ckpt_freq == 0:
            multi_agent.save(ckpt_dir, episode)
            logger.info("Checkpoint saved at episode %d", episode)

    # Final save
    multi_agent.save(ckpt_dir, n_episodes)
    tlog.close()
    logger.info("Training complete.")
    return multi_agent


#  CLI 

def _parse_args():
    p = argparse.ArgumentParser(description="Train SUMOFlow AI DQN agents")
    p.add_argument("--config",   default="DeepQN/configs/dqn_config.yaml",
                   help="Path to dqn_config.yaml")
    p.add_argument("--profile",  default=None,
                   help="Force a single traffic profile (default: rotate all 4)")
    p.add_argument("--episodes", type=int, default=None,
                   help="Override number of training episodes")
    p.add_argument("--resume",   default=None,
                   help="Checkpoint directory to resume from")
    p.add_argument("--port",     type=int, default=8813,
                   help="TraCI TCP port")
    p.add_argument("--gui",      action="store_true",
                   help="Launch sumo-gui instead of headless sumo")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.gui:
        # Patch config to use sumo-gui
        import yaml
        with open(args.config) as f:
            cfg_raw = yaml.safe_load(f)
        cfg_raw["simulation"]["sumo_binary"] = "sumo-gui"
        import tempfile
        tmp = tempfile.NamedTemporaryFile(
            suffix=".yaml", mode="w", delete=False
        )
        yaml.dump(cfg_raw, tmp)
        tmp.close()
        args.config = tmp.name

    train(
        config_path      = args.config,
        override_profile = args.profile,
        num_episodes     = args.episodes,
        resume_dir       = args.resume,
        port             = args.port,
    )
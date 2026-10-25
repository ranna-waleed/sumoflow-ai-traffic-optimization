from __future__ import annotations

import copy
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import torch

from DeepQN.agent.dqn_agent import CheckpointValidationError, MultiAgentDQN


CFG = {
    "state_dim": 3,
    "action_dim": 2,
    "hidden_dims": [4],
    "batch_size": 2,
    "min_replay_size": 2,
}


class StrictCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.agents = MultiAgentDQN(["tls_a", "tls_b"], CFG)

    def tearDown(self):
        self.temp.cleanup()

    def _save(self, tls_id: str, episode: int, *, recorded_episode=None, incompatible=False):
        agent = self.agents.agents[tls_id]
        online = copy.deepcopy(agent.online_net.state_dict())
        target = copy.deepcopy(agent.target_net.state_dict())
        if incompatible:
            first = next(iter(online))
            online[first] = torch.zeros(1)
        torch.save(
            {
                "online": online,
                "target": target,
                "episode": episode if recorded_episode is None else recorded_episode,
                "epsilon": 0.1,
                "steps": 12,
            },
            self.root / f"{tls_id}_ep{episode:04d}.pt",
        )

    def test_valid_complete_ensemble_loads_and_hashes_every_agent(self):
        self._save("tls_a", 5)
        self._save("tls_b", 5)
        manifest = self.agents.load_for_evaluation(self.root, 5)
        self.assertEqual(manifest.episode, 5)
        self.assertEqual({item.tls_id for item in manifest.checkpoints}, {"tls_a", "tls_b"})
        self.assertTrue(all(len(item.sha256) == 64 for item in manifest.checkpoints))
        self.assertTrue(all(self.root.resolve() in Path(item.path).parents for item in manifest.checkpoints))

    def test_evaluation_uses_restricted_checkpoint_loading(self):
        self._save("tls_a", 5); self._save("tls_b", 5)
        real_load = torch.load
        calls = []
        def recording_load(*args, **kwargs):
            calls.append(kwargs.get("weights_only"))
            return real_load(*args, **kwargs)
        with patch("DeepQN.agent.dqn_agent.torch.load", side_effect=recording_load):
            self.agents.load_for_evaluation(self.root, 5)
        self.assertEqual(calls, [True, True])

    def test_missing_agent_is_rejected(self):
        self._save("tls_a", 5)
        with self.assertRaisesRegex(CheckpointValidationError, "missing checkpoint"):
            self.agents.load_for_evaluation(self.root, 5)

    def test_missing_directory_is_rejected_for_evaluation(self):
        with self.assertRaisesRegex(CheckpointValidationError, "directory not found"):
            self.agents.load_for_evaluation(self.root / "not-present", 5)

    def test_inconsistent_recorded_episode_is_rejected(self):
        self._save("tls_a", 5)
        self._save("tls_b", 5, recorded_episode=4)
        with self.assertRaisesRegex(CheckpointValidationError, "records episode"):
            self.agents.load_for_evaluation(self.root, 5)

    def test_incompatible_weights_are_rejected_atomically(self):
        before = {
            tls_id: {key: value.clone() for key, value in agent.online_net.state_dict().items()}
            for tls_id, agent in self.agents.agents.items()
        }
        self._save("tls_a", 5)
        self._save("tls_b", 5, incompatible=True)
        with self.assertRaisesRegex(CheckpointValidationError, "incompatible"):
            self.agents.load_for_evaluation(self.root, 5)
        for tls_id, agent in self.agents.agents.items():
            for key, value in agent.online_net.state_dict().items():
                self.assertTrue(torch.equal(value, before[tls_id][key]))

    def test_malformed_checkpoint_structure_is_rejected(self):
        torch.save({"episode": 5}, self.root / "tls_a_ep0005.pt")
        self._save("tls_b", 5)
        with self.assertRaisesRegex(CheckpointValidationError, "missing keys"):
            self.agents.load_for_evaluation(self.root, 5)

    def test_training_resume_still_allows_absent_directory(self):
        self.agents.load_latest(self.root / "not-present")


if __name__ == "__main__":
    unittest.main()

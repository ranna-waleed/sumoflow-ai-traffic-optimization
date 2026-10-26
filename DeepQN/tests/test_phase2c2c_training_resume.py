from __future__ import annotations

import copy
import hashlib
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from DeepQN.agent.dqn_agent import MultiAgentDQN
from DeepQN.training.research_resume import (
    ResearchResumeError, ResumeContracts, TrainingCursor,
    publish_training_snapshot, restore_training_snapshot,
)
from DeepQN.training.research_orchestration import (
    EpisodeResult, ResearchSchedule, plan_next_episode,
    run_research_episode_transactions,
)
from DeepQN.training.train import require_research_training_authority, train


def _cfg(per=False):
    return {"state_dim": 4, "action_dim": 2, "hidden_dims": [8],
            "learning_rate": 0.001, "replay_buffer_size": 32,
            "min_replay_size": 2, "batch_size": 2, "gamma": 0.9,
            "epsilon_start": 0.8, "epsilon_end": 0.1, "epsilon_decay": 0.95,
            "tau": 0.2, "seed": 17, "prioritised_replay": per}


def _contracts(**changes):
    values = dict(run_id="synthetic-run", run_manifest_sha256="1" * 64,
                  dataset_manifest_sha256="2" * 64, demand_registry_sha256="3" * 64,
                  split_registry_sha256="4" * 64, observation_contract="obs-v1",
                  reward_contract="reward-v1", forecast_contract="no-forecast-v1",
                  git_commit="abc", environment_sha256="5" * 64)
    values.update(changes)
    return ResumeContracts(**values)


def _step(ensemble, n):
    trace = []
    for index in range(n):
        state = np.array([index, index + 1, -index, 0.5], dtype=np.float32) / 10
        action = ensemble.act({"tls": state})["tls"]
        ensemble.push({"tls": state}, {"tls": action}, {"tls": float(index % 3)},
                      {"tls": state + 0.1}, terminated=False, interval_duration_s=10)
        loss = ensemble.update()["tls"]
        trace.append((action, ensemble.agents["tls"].epsilon, loss))
    return trace


def _semantic(agent):
    replay = agent.replay.state_dict()
    return {"online": {k: v.detach().clone() for k, v in agent.online_net.state_dict().items()},
            "target": {k: v.detach().clone() for k, v in agent.target_net.state_dict().items()},
            "optimizer": copy.deepcopy(agent.optimiser.state_dict()), "replay": replay,
            "epsilon": agent.epsilon, "steps": agent.total_steps, "updates": agent.total_updates}


def _rehash_snapshot(root: Path) -> str:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    global_path = root / manifest["global_state"]["filename"]
    manifest["global_state"]["sha256"] = hashlib.sha256(global_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (root / "COMPLETE.json").write_text(json.dumps({
        "schema": "sumoflow-research-training-complete-v1", "manifest_sha256": digest
    }, sort_keys=True) + "\n", encoding="utf-8")
    return digest


class TrainingResumeTests(unittest.TestCase):
    def setUp(self):
        random.seed(9); np.random.seed(9); torch.manual_seed(9)

    def _round_trip(self, per):
        ensemble = MultiAgentDQN(["tls"], _cfg(per)); _step(ensemble, 4)
        with tempfile.TemporaryDirectory() as temp:
            result = publish_training_snapshot(temp, ensemble, TrainingCursor(1, 2, 1, 2, 3, 104), _contracts())
            expected_next = _step(ensemble, 4)
            expected = _semantic(ensemble.agents["tls"])
            # Fresh construction consumes RNG; restoration must put it back.
            random.random(); np.random.random(); torch.rand(3)
            restored = MultiAgentDQN(["tls"], _cfg(per))
            cursor = restore_training_snapshot(result["path"], restored, _contracts(), result["manifest_sha256"])
            actual_next = _step(restored, 4)
            actual = _semantic(restored.agents["tls"])
            self.assertEqual(cursor.next_episode, 2)
            self.assertEqual(expected_next, actual_next)
            self.assertEqual(expected["epsilon"], actual["epsilon"])
            self.assertEqual(expected["steps"], actual["steps"])
            self.assertEqual(expected["updates"], actual["updates"])
            for key in expected["online"]:
                self.assertTrue(torch.equal(expected["online"][key], actual["online"][key]))
                self.assertTrue(torch.equal(expected["target"][key], actual["target"][key]))
            if per:
                np.testing.assert_array_equal(expected["replay"]["priorities"], actual["replay"]["priorities"])
                self.assertEqual(expected["replay"]["ptr"], actual["replay"]["ptr"])
                self.assertEqual(expected["replay"]["beta"], actual["replay"]["beta"])

    def test_uniform_interruption_equivalence(self): self._round_trip(False)
    def test_per_interruption_equivalence(self): self._round_trip(True)

    def test_requires_complete_episode_boundary(self):
        with self.assertRaises(ValueError): TrainingCursor(1, 2, 0, 0, 0, 1, False).validate()
        with self.assertRaises(ValueError): TrainingCursor(1, 3, 0, 0, 0, 1).validate()

    def test_publication_failure_leaves_no_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(OSError):
                publish_training_snapshot(temp, MultiAgentDQN(["a", "b"], _cfg()),
                                          TrainingCursor(1, 2, 0, 0, 0, 1), _contracts(),
                                          fail_after_files=1)
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_missing_or_modified_file_and_marker_rejected(self):
        for mutation in ("agent", "marker"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                source = MultiAgentDQN(["tls"], _cfg())
                result = publish_training_snapshot(temp, source, TrainingCursor(1, 2, 0, 0, 0, 1), _contracts())
                root = Path(result["path"])
                if mutation == "agent": (root / "agent-tls.pt").write_bytes(b"altered")
                else: (root / "COMPLETE.json").unlink()
                target = MultiAgentDQN(["tls"], _cfg()); before = _semantic(target.agents["tls"])
                with self.assertRaises(ResearchResumeError):
                    restore_training_snapshot(root, target, _contracts(), result["manifest_sha256"])
                for key in before["online"]:
                    self.assertTrue(torch.equal(before["online"][key], target.agents["tls"].online_net.state_dict()[key]))

    def test_contract_cursor_and_ensemble_mismatches_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            result = publish_training_snapshot(temp, MultiAgentDQN(["tls"], _cfg()),
                TrainingCursor(7, 8, 2, 3, 4, 99), _contracts())
            for target, contract in ((MultiAgentDQN(["tls"], _cfg()), _contracts(demand_registry_sha256="9" * 64)),
                                     (MultiAgentDQN(["other"], _cfg()), _contracts())):
                with self.assertRaises(ResearchResumeError):
                    restore_training_snapshot(result["path"], target, contract, result["manifest_sha256"])

    def test_observation_reward_and_forecast_contract_mismatches_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            result = publish_training_snapshot(temp, MultiAgentDQN(["tls"], _cfg()),
                TrainingCursor(2, 3, 0, 0, 0, 7), _contracts())
            for changes in ({"observation_contract": "obs-v2"}, {"reward_contract": "reward-v2"},
                            {"forecast_contract": "model-v2"}):
                with self.subTest(changes=changes), self.assertRaises(ResearchResumeError):
                    restore_training_snapshot(result["path"], MultiAgentDQN(["tls"], _cfg()),
                                              _contracts(**changes), result["manifest_sha256"])

    def test_self_consistent_agent_tampering_still_fails_identity_validation(self):
        """A same-source rehash cannot bypass episode/TLS/schema validation."""
        for field, value in (("episode", 99), ("tls_id", "other"), ("epsilon", "invalid")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                result = publish_training_snapshot(temp, MultiAgentDQN(["tls"], _cfg()),
                    TrainingCursor(2, 3, 0, 0, 0, 7), _contracts())
                root = Path(result["path"]); agent_path = root / "agent-tls.pt"
                payload = torch.load(agent_path, weights_only=True); payload[field] = value
                torch.save(payload, agent_path)
                manifest_path = root / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["agents"][0]["sha256"] = hashlib.sha256(agent_path.read_bytes()).hexdigest()
                manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                (root / "COMPLETE.json").write_text(json.dumps({
                    "schema": "sumoflow-research-training-complete-v1",
                    "manifest_sha256": manifest_hash}, sort_keys=True) + "\n", encoding="utf-8")
                target = MultiAgentDQN(["tls"], _cfg()); before = _semantic(target.agents["tls"])
                with self.assertRaises(ResearchResumeError):
                    restore_training_snapshot(root, target, _contracts(), manifest_hash)
                for key in before["online"]:
                    self.assertTrue(torch.equal(before["online"][key], target.agents["tls"].online_net.state_dict()[key]))

    def test_external_manifest_hash_is_required(self):
        with tempfile.TemporaryDirectory() as temp:
            result = publish_training_snapshot(temp, MultiAgentDQN(["tls"], _cfg()),
                TrainingCursor(0, 1, 0, 0, 0, 1), _contracts())
            with self.assertRaises(ResearchResumeError):
                restore_training_snapshot(result["path"], MultiAgentDQN(["tls"], _cfg()), _contracts(), "0" * 64)

    def test_non_overwrite_and_research_entrypoint_block(self):
        with tempfile.TemporaryDirectory() as temp:
            args = (temp, MultiAgentDQN(["tls"], _cfg()), TrainingCursor(0, 1, 0, 0, 0, 1), _contracts())
            publish_training_snapshot(*args)
            with self.assertRaises(FileExistsError): publish_training_snapshot(*args)
        with self.assertRaisesRegex(RuntimeError, "trust roots are absent"):
            require_research_training_authority()
        with self.assertRaisesRegex(RuntimeError, "trust roots are absent"):
            train(config_path="does-not-exist.yaml", research_mode=True)

    def test_cuda_requirement_is_derived_and_cannot_be_overridden(self):
        with tempfile.TemporaryDirectory() as temp:
            ensemble = MultiAgentDQN(["tls"], _cfg())
            with self.assertRaisesRegex(ResearchResumeError, "contradicts actual agent devices"):
                publish_training_snapshot(temp, ensemble, TrainingCursor(0, 1, 0, 0, 0, 1),
                                          _contracts(), cuda_required=True)

    def test_inconsistent_cuda_metadata_is_rejected_on_cpu_without_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            source = MultiAgentDQN(["tls"], _cfg())
            result = publish_training_snapshot(temp, source, TrainingCursor(0, 1, 0, 0, 0, 1), _contracts())
            root = Path(result["path"]); global_path = root / "global-state.pt"
            payload = torch.load(global_path, weights_only=True)
            payload["cuda_required"] = True; payload["cuda_device_count"] = 1
            payload["cuda_device_names"] = ["fabricated-gpu"]; payload["torch_cuda_rng"] = []
            torch.save(payload, global_path); manifest_hash = _rehash_snapshot(root)
            target = MultiAgentDQN(["tls"], _cfg()); before = _semantic(target.agents["tls"])
            py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
            with self.assertRaisesRegex(ResearchResumeError, "CUDA requirement"):
                restore_training_snapshot(root, target, _contracts(), manifest_hash)
            for key in before["online"]:
                self.assertTrue(torch.equal(before["online"][key], target.agents["tls"].online_net.state_dict()[key]))
            self.assertEqual(py_state, random.getstate())
            np.testing.assert_array_equal(np_state[1], np.random.get_state()[1])
            self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))

    def test_episode_orchestration_advances_only_after_finalization(self):
        schedule = ResearchSchedule(("p0", "p1"), ("d0", "d1"), (10, 20, 30))
        cursor = TrainingCursor(0, 1, 0, 0, 0, 10); plans = []
        with tempfile.TemporaryDirectory() as temp:
            ensemble = MultiAgentDQN(["tls"], _cfg())
            def successful(plan, _agents):
                plans.append(plan)
                return EpisodeResult(True, True, {"identity": plan.episode})
            cursor, records = run_research_episode_transactions(
                agents=ensemble, initial_cursor=cursor, schedule=schedule,
                contracts=_contracts(), snapshot_root=temp, episode_runner=successful,
                episode_count=2)
            self.assertEqual([(p.episode, p.profile, p.demand_realization_id, p.sumo_seed)
                              for p in plans], [(1, "p0", "d0", 10), (2, "p1", "d1", 20)])
            self.assertEqual((cursor.next_episode, cursor.profile_cursor, cursor.demand_cursor,
                              cursor.sumo_run_index, cursor.next_sumo_seed), (3, 0, 0, 2, 30))
            self.assertEqual(len(records), 2)
            before = sorted(Path(temp).iterdir())
            with self.assertRaisesRegex(RuntimeError, "cursor was not advanced"):
                run_research_episode_transactions(agents=ensemble, initial_cursor=cursor,
                    schedule=schedule, contracts=_contracts(), snapshot_root=temp,
                    episode_runner=lambda *_: EpisodeResult(False, False, {}), episode_count=1)
            self.assertEqual(before, sorted(Path(temp).iterdir()))

    def test_schedule_rejects_corrupted_profile_demand_and_seed_cursors(self):
        schedule = ResearchSchedule(("p",), ("d",), (11,))
        for cursor in (TrainingCursor(0, 1, 1, 0, 0, 11),
                       TrainingCursor(0, 1, 0, 1, 0, 11),
                       TrainingCursor(0, 1, 0, 0, 0, 12)):
            with self.subTest(cursor=cursor), self.assertRaises(ValueError):
                plan_next_episode(cursor, schedule)

    def test_fresh_process_equivalence_uniform_and_per(self):
        for per in (False, True):
            with self.subTest(per=per), tempfile.TemporaryDirectory() as full_dir, tempfile.TemporaryDirectory() as resumed_dir:
                suffix = ["--per"] if per else []
                commands = [
                    [sys.executable, "-m", "DeepQN.training.resume_probe", "full", full_dir, *suffix],
                    [sys.executable, "-m", "DeepQN.training.resume_probe", "save", resumed_dir, *suffix],
                    [sys.executable, "-m", "DeepQN.training.resume_probe", "resume", resumed_dir, *suffix],
                ]
                for command in commands:
                    subprocess.run(command, check=True, capture_output=True, text=True)
                full = json.loads((Path(full_dir) / "full.json").read_text(encoding="utf-8"))
                first = json.loads((Path(resumed_dir) / "save.json").read_text(encoding="utf-8"))
                second = json.loads((Path(resumed_dir) / "resume.json").read_text(encoding="utf-8"))
                self.assertEqual(full["trace"], first["trace"] + second["trace"])
                self.assertEqual(full["digest"], second["digest"])
                self.assertEqual(full["records"], first["records"] + second["records"])
                self.assertEqual(full["cursor"], second["cursor"])


if __name__ == "__main__":
    unittest.main()

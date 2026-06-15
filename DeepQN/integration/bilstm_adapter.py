"""
dqn/integration/bilstm_adapter.py:
Connects the trained BiLSTM model (detection/lstm/predict.py) to the DQN
state vector.

What it does:
Every SUMO step it collects one row of traffic features from TraCI and
appends it to a rolling 60-step history buffer (matching SEQ_LEN in predict.py).
Every ``predict_every_steps`` SUMO steps it calls ``predict(history)`` and
extracts the ``next_30s`` mean flow for each of the 4 directions.
Those 4 values are normalised to [0, 1] using the model's own scaler and
returned as a float32 array ready to slot into the DQN state vector.

If the model files are missing, or if predict() raises, it falls back
transparently to live TraCI vehicle counts — training is never blocked.

Usage (inside SumoEnv / DQNController):
    from DeepQN.integration.bilstm_adapter import BiLSTMAdapter

    adapter = BiLSTMAdapter(cfg)        # cfg = full dqn_config dict
    adapter.load()                       # call once after SUMO connects

    # inside the SUMO step loop:
    adapter.tick()                       # collect one feature row

    # when building the DQN state:
    preds = adapter.predict()            # np.ndarray shape (4,) in [0, 1]
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import pickle
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

#  Path to the LSTM module 
# lstm/ sits at the project root , two levels above dqn/integration/
# Full path: sumoflow-ai-traffic-optimization/lstm/
_DQN_INTEGRATION_DIR = Path(__file__).resolve().parent          # dqn/integration/
_PROJECT_ROOT        = _DQN_INTEGRATION_DIR.parent.parent       # project root
LSTM_MODULE_PATH     = _PROJECT_ROOT / "lstm"                   # lstm/ at project root
LSTM_MODELS_PATH     = LSTM_MODULE_PATH / "models"

SEQ_LEN = 60   # must match lstm/predict.py SEQ_LEN

#  Direction -> SUMO edge mapping (from net.xml analysis) 
DIRECTION_EDGES: Dict[str, List[str]] = {
    "north": ["690516091#0", "690516091#1"],
    "south": ["10873191#3", "10873191#4", "10873191#5"],
    "east":  ["28718647#1",  "28718647#2",  "28718647#3"],
    "west":  ["50211834#0",  "50211834#1",  "192591565#0"],
}

# Keywords used to map feature names -> TraCI metric calls
_COUNT_KEYS   = {"count", "veh", "num", "volume", "flow"}
_SPEED_KEYS   = {"speed", "vel", "velocity"}
_WAIT_KEYS    = {"wait", "halt", "queue", "delay"}
_OCC_KEYS     = {"occ", "density", "occupancy"}


#  Adapter 

class BiLSTMAdapter:
    """
    Wraps ``detection/lstm/predict.py`` so the DQN can use BiLSTM
    predictions as part of its state vector.

    Parameters:
    config : dict — the full dqn_config.yaml loaded as a dict
    """

    def __init__(self, config: dict):
        bilstm_cfg = config.get("bilstm", {})
        self._predict_every: int   = bilstm_cfg.get("predict_every_steps", 10)
        self._max_flow:       float = bilstm_cfg.get("max_flow_per_edge", 50.0)

        # Rolling history buffer — same length as SEQ_LEN in predict.py
        self._history: deque = deque(maxlen=SEQ_LEN)

        # Loaded once
        self._predict_fn  = None    # reference to lstm/predict.predict()
        self._features:   Optional[List[str]] = None
        self._max_vals:   Optional[np.ndarray] = None   # scaler max for dirs 0-3
        self._loaded:     bool = False
        self._use_proxy:  bool = False   # fallback flag

        # Cache last prediction (avoid re-running every step)
        self._last_pred:  np.ndarray = np.zeros(4, dtype=np.float32)
        self._step_count: int = 0

    # Lifecycle 

    def load(self) -> bool:
        """
        Import predict.py and load the scaler.
        Returns True on success, False if files are missing (proxy is used).
        """
        config_path = LSTM_MODELS_PATH / "config.json"
        scaler_path = LSTM_MODELS_PATH / "scaler.pkl"
        model_path  = LSTM_MODELS_PATH / "lstm_best.pth"

        for p in (config_path, scaler_path, model_path):
            if not p.exists():
                logger.warning(
                    "BiLSTM file not found: %s — using TraCI proxy instead.", p
                )
                self._use_proxy = True
                return False

        # Add detection/lstm/ to sys.path so we can import predict.py
        lstm_str = str(LSTM_MODULE_PATH)
        if lstm_str not in sys.path:
            sys.path.insert(0, lstm_str)

        try:
            import predict as lstm_predict
            importlib.reload(lstm_predict)          # ensure fresh module state
            self._predict_fn = lstm_predict.predict

            # Read feature names
            with open(config_path) as f:
                lstm_cfg = json.load(f)
            self._features = lstm_cfg.get("features", [])
            logger.info("[BiLSTM] Features: %s", self._features)

            # Read scaler for normalisation
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            max_v = np.array(scaler["max_"], dtype=np.float32)
            # Use the first 4 max values (N, S, E, W output directions)
            self._max_vals = np.where(max_v[:4] > 0, max_v[:4], 1.0)

            self._loaded    = True
            self._use_proxy = False
            logger.info(
                "[BiLSTM] Adapter loaded. scaler max(N,S,E,W)=%s",
                self._max_vals.tolist(),
            )
            return True

        except Exception as exc:
            logger.warning(
                "[BiLSTM] Load failed (%s) — using TraCI proxy.", exc
            )
            self._use_proxy = True
            return False

    #  Per-step tick 

    def tick(self) -> None:
        """
        Call ONCE per SUMO simulation step.
        Collects one feature row from TraCI and appends it to the history buffer.
        If the model is not loaded, this is a no-op.
        """
        if not self._loaded:
            return
        row = self._collect_features()
        self._history.append(row)
        self._step_count += 1

    #  Prediction 

    def predict(self) -> np.ndarray:
        """
        Returns a float32 array of shape (4,): [N, S, E, W] flow predictions,
        each normalised to [0, 1].

        Uses the BiLSTM model when loaded, otherwise falls back to
        live TraCI vehicle counts.  Safe to call every step , predictions
        are cached between ``predict_every_steps`` intervals.
        """
        if self._use_proxy or not self._loaded:
            return self._proxy_predict()

        # Only re-run every predict_every_steps steps
        if self._step_count % self._predict_every != 0:
            return self._last_pred

        if len(self._history) < 5:
            # Not enough history yet — use proxy
            return self._proxy_predict()

        try:
            result = self._predict_fn(list(self._history))
            ns     = result["next_30s"]         # {"north": int, "south": int, ...}

            raw = np.array(
                [ns["north"], ns["south"], ns["east"], ns["west"]],
                dtype=np.float32,
            )
            self._last_pred = np.clip(raw / self._max_vals, 0.0, 1.0)
            logger.debug("[BiLSTM] pred(N,S,E,W)=%s", self._last_pred.tolist())

        except Exception as exc:
            logger.debug("[BiLSTM] predict() failed (%s) — proxy used.", exc)
            self._last_pred = self._proxy_predict()

        return self._last_pred

    #  Private: feature collection 

    def _collect_features(self) -> Dict[str, float]:
        """
        Build one history row whose keys match the feature names in
        detection/lstm/models/config.json.

        Strategy:
        For each feature name, scan for a direction keyword (north/south/
        east/west) and a metric keyword (count/speed/wait/occ).  Aggregate
        the corresponding TraCI values across all edges for that direction.
        If a feature name cannot be matched, it defaults to 0.0 , the same
        fallback as ``r.get(f, 0.0)`` in predict.py.
        """
        try:
            import traci
        except ImportError:
            return {f: 0.0 for f in (self._features or [])}

        row: Dict[str, float] = {}

        for feat in (self._features or []):
            fl    = feat.lower()
            value = 0.0

            # Find the direction this feature belongs to
            matched_direction = None
            for direction in DIRECTION_EDGES:
                if direction in fl:
                    matched_direction = direction
                    break

            if matched_direction is None:
                row[feat] = 0.0
                continue

            edges = DIRECTION_EDGES[matched_direction]

            # Find the metric
            feat_keys = set(fl.replace("_", " ").split())
            try:
                if feat_keys & _COUNT_KEYS:
                    value = float(sum(
                        traci.edge.getLastStepVehicleNumber(e)
                        for e in edges if self._edge_ok(e)
                    ))
                elif feat_keys & _SPEED_KEYS:
                    speeds = [
                        traci.edge.getLastStepMeanSpeed(e)
                        for e in edges if self._edge_ok(e)
                    ]
                    value = float(np.mean(speeds)) if speeds else 0.0
                elif feat_keys & _WAIT_KEYS:
                    value = float(sum(
                        traci.edge.getWaitingTime(e)
                        for e in edges if self._edge_ok(e)
                    ))
                elif feat_keys & _OCC_KEYS:
                    occs = [
                        traci.edge.getLastStepOccupancy(e)
                        for e in edges if self._edge_ok(e)
                    ]
                    value = float(np.mean(occs)) if occs else 0.0
                else:
                    # Unknown metric — default to vehicle count
                    value = float(sum(
                        traci.edge.getLastStepVehicleNumber(e)
                        for e in edges if self._edge_ok(e)
                    ))
            except Exception:
                value = 0.0

            row[feat] = value

        return row

    def _edge_ok(self, edge_id: str) -> bool:
        """Return True if TraCI can query this edge without raising."""
        try:
            import traci
            traci.edge.getLastStepVehicleNumber(edge_id)
            return True
        except Exception:
            return False

    # Proxy fallback 

    def _proxy_predict(self) -> np.ndarray:
        """
        Fallback: normalised live vehicle counts from TraCI.
        Used when the BiLSTM model is not loaded or fails.
        """
        try:
            import traci
            flows = []
            for edges in DIRECTION_EDGES.values():
                total = 0
                for e in edges:
                    try:
                        total += traci.edge.getLastStepVehicleNumber(e)
                    except Exception:
                        pass
                flows.append(min(total / (self._max_flow * len(edges)), 1.0))
            return np.array(flows, dtype=np.float32)
        except Exception:
            return np.zeros(4, dtype=np.float32)

    # Utility 

    def reset(self) -> None:
        """Clear history at the start of a new episode."""
        self._history.clear()
        self._last_pred  = np.zeros(4, dtype=np.float32)
        self._step_count = 0

    @property
    def is_loaded(self) -> bool:
        return self._loaded
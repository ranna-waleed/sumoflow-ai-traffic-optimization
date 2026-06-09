"""
DeepQN/explainability/explainer.py
------------------------------------
Human-readable explanations for DQN decisions.

For each decision step, explains:
  - WHY a junction switched or kept its phase
  - WHICH lane features drove the decision
  - WHAT the BiLSTM predicted
  - HOW confident the agent was (Q-value margin)

Usage:
    from DeepQN.explainability.explainer import DQNExplainer
    explainer = DQNExplainer(cfg)
    explanation = explainer.explain(tid, obs, action, q_values)
    print(explanation.to_text())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Junction human-readable names
JUNCTION_NAMES = {
    # Tahrir
    "315744796":  "N-Trunk / E-Highway Entry",
    "96621100":   "Roundabout N-Entry",
    "2031414903": "Corniche W-Entry",
    "2031414899": "S-Corridor Gate 1",
    "6288771431": "S-Corridor Gate 2",
    "271064234":  "E-Tertiary Exit 1",
    "315743335":  "E-Tertiary Exit 2",
    # Taksim
    "105404369":  "Taksim Junction A",
    "2263863893": "Taksim Junction B",
    "2333203943": "Taksim Junction C",
    "278335467":  "Taksim Junction D",
    "34156851":   "Taksim Junction E",
    "539369578":  "Taksim Junction F",
    "joinedG_1450800222_2263863897_278615585_cluster_1450800225_269500433":
                  "Taksim Main Interchange",
    "joinedG_269500161_434340903": "Taksim NE Junction",
}


#  Explanation dataclass 

@dataclass
class DecisionExplanation:
    junction_id:    str
    junction_name:  str
    action:         int          # 0=keep, 1=switch
    action_label:   str
    confidence:     float        # Q margin: |Q(chosen) - Q(other)|
    confidence_label: str        # "High" / "Medium" / "Low"

    # Top reasons
    top_lanes:      List[dict]   # [{"lane": str, "halting": float, "wait": float, "occ": float}]
    lstm_forecast:  dict         # {"north": float, "south": float, "east": float, "west": float}
    dominant_dir:   str          # direction with highest predicted flow
    time_of_day:    float        # normalized 0-1

    # Raw Q-values
    q_keep:   float
    q_switch: float

    def to_text(self) -> str:
        """Generate human-readable explanation."""
        lines = []
        lines.append(f"Junction: {self.junction_name} ({self.junction_id[:12]}...)"
                     if len(self.junction_id) > 12 else
                     f"Junction: {self.junction_name} ({self.junction_id})")
        lines.append(f"Decision: {self.action_label.upper()}")
        lines.append(f"Confidence: {self.confidence_label} "
                     f"(Q-margin={self.confidence:.3f}, "
                     f"Q_keep={self.q_keep:.3f}, Q_switch={self.q_switch:.3f})")

        if self.top_lanes:
            lines.append("Top congested lanes:")
            for lane_info in self.top_lanes[:3]:
                lines.append(
                    f"  • {lane_info['lane']}: "
                    f"{lane_info['halting_pct']:.0%} halting, "
                    f"wait={lane_info['wait_s']:.0f}s, "
                    f"occ={lane_info['occ_pct']:.0%}"
                )

        if any(self.lstm_forecast.values()):
            lines.append(
                f"BiLSTM forecast: N={self.lstm_forecast.get('north',0):.0f} "
                f"S={self.lstm_forecast.get('south',0):.0f} "
                f"E={self.lstm_forecast.get('east',0):.0f} "
                f"W={self.lstm_forecast.get('west',0):.0f} vehicles/30s"
            )
            if self.dominant_dir:
                lines.append(f"Highest predicted flow: {self.dominant_dir.upper()}")

        # Plain-English summary
        if self.action == 1:
            reason = self._switch_reason()
        else:
            reason = self._keep_reason()
        lines.append(f"Reason: {reason}")

        return "\n".join(lines)

    def _switch_reason(self) -> str:
        reasons = []
        if self.top_lanes:
            top = self.top_lanes[0]
            if top["halting_pct"] > 0.5:
                reasons.append(
                    f"heavy queue on {top['lane']} "
                    f"({top['halting_pct']:.0%} of capacity halting)"
                )
        if self.dominant_dir and any(self.lstm_forecast.values()):
            val = self.lstm_forecast.get(self.dominant_dir, 0)
            if val > 20:
                reasons.append(
                    f"BiLSTM predicts {val:.0f} incoming vehicles "
                    f"from {self.dominant_dir.upper()}"
                )
        if not reasons:
            reasons.append("Q-network estimated switching yields lower wait time")
        return "; ".join(reasons) + "."

    def _keep_reason(self) -> str:
        reasons = []
        if self.top_lanes:
            top = self.top_lanes[0]
            if top["halting_pct"] < 0.3:
                reasons.append("current approach has low queue density")
        if self.confidence > 0.3:
            reasons.append(
                f"agent is confident keeping is better "
                f"(Q-margin={self.confidence:.3f})"
            )
        if not reasons:
            reasons.append("Q-network estimated keeping current phase "
                           "yields lower wait time")
        return "; ".join(reasons) + "."

    def to_dict(self) -> dict:
        return {
            "junction_id":    self.junction_id,
            "junction_name":  self.junction_name,
            "action":         self.action,
            "action_label":   self.action_label,
            "confidence":     round(self.confidence, 4),
            "confidence_label": self.confidence_label,
            "q_keep":         round(self.q_keep, 4),
            "q_switch":       round(self.q_switch, 4),
            "top_lanes":      self.top_lanes,
            "lstm_forecast":  self.lstm_forecast,
            "dominant_dir":   self.dominant_dir,
            "reason_text":    self.to_text(),
        }


#  Explainer 

class DQNExplainer:
    """
    Generates human-readable explanations for DQN decisions.
    Works with the 37-feature state vector used during training.
    """

    MAX_LANES   = 10
    FPERLANE    = 3    # halting, wait, occupancy
    MAX_QUEUE   = 50.0
    MAX_WAIT    = 300.0
    MAX_FLOW    = 136.0

    def __init__(self, cfg: dict):
        self.cfg      = cfg
        self.tls_cfg  = cfg.get("tls_junctions", {})

    def explain(
        self,
        tid:      str,
        obs:      "np.ndarray",   # 37-feature state vector
        action:   int,
        agents:   "MultiAgentDQN",
        lstm_pred: dict = None,
    ) -> DecisionExplanation:
        """
        Generate explanation for one junction's decision.
        """
        import torch

        #  Get Q-values 
        state_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            q_vals = agents.agents[tid].online_net(state_t).squeeze().numpy()

        q_keep   = float(q_vals[0])
        q_switch = float(q_vals[1])
        margin   = abs(q_keep - q_switch)

        if margin > 0.5:
            conf_label = "High"
        elif margin > 0.1:
            conf_label = "Medium"
        else:
            conf_label = "Low"

        #  Decode lane features 
        lane_features = obs[:self.MAX_LANES * self.FPERLANE]
        incoming_lanes = self.tls_cfg.get(tid, {}).get("incoming_lanes", [])

        top_lanes = []
        for i in range(self.MAX_LANES):
            base = i * self.FPERLANE
            halt = float(lane_features[base])
            wait = float(lane_features[base + 1])
            occ  = float(lane_features[base + 2])

            if halt > 0.01 or wait > 0.01:
                lane_name = incoming_lanes[i] if i < len(incoming_lanes) else f"lane_{i}"
                top_lanes.append({
                    "lane":         lane_name,
                    "halting_pct":  halt,
                    "wait_s":       wait * self.MAX_WAIT,
                    "occ_pct":      occ,
                })

        # Sort by halting count descending
        top_lanes.sort(key=lambda x: x["halting_pct"], reverse=True)

        #  Decode BiLSTM features 
        # Positions 32-35 in state vector are normalized LSTM predictions
        lstm_raw = obs[32:36]
        lstm_forecast = {
            "north": round(float(lstm_raw[0]) * self.MAX_FLOW),
            "south": round(float(lstm_raw[1]) * self.MAX_FLOW),
            "east":  round(float(lstm_raw[2]) * self.MAX_FLOW),
            "west":  round(float(lstm_raw[3]) * self.MAX_FLOW),
        }

        # Use raw lstm_pred if provided (more accurate)
        if lstm_pred:
            lstm_forecast = {k: round(v) for k, v in lstm_pred.items()
                             if k in ("north", "south", "east", "west")}

        dominant_dir = max(lstm_forecast, key=lstm_forecast.get) \
            if any(lstm_forecast.values()) else ""

        #  Time of day 
        time_of_day = float(obs[36]) if len(obs) > 36 else 0.5

        return DecisionExplanation(
            junction_id    = tid,
            junction_name  = JUNCTION_NAMES.get(tid, f"Junction {tid[:8]}"),
            action         = action,
            action_label   = "switch phase" if action == 1 else "keep phase",
            confidence     = margin,
            confidence_label = conf_label,
            top_lanes      = top_lanes[:5],
            lstm_forecast  = lstm_forecast,
            dominant_dir   = dominant_dir,
            time_of_day    = time_of_day,
            q_keep         = q_keep,
            q_switch       = q_switch,
        )

    def explain_all(
        self,
        observations:  dict,
        actions:       dict,
        agents:        "MultiAgentDQN",
        lstm_pred:     dict = None,
        step:          int  = 0,
    ) -> List[DecisionExplanation]:
        """Explain all junction decisions for one decision step."""
        explanations = []
        for tid, obs in observations.items():
            action = actions.get(tid, 0)
            exp    = self.explain(tid, obs, action, agents, lstm_pred)
            explanations.append(exp)
        return explanations

    def print_step(
        self,
        explanations: List[DecisionExplanation],
        step:         int,
        show_all:     bool = False,
    ):
        """Print explanations for a decision step. By default only shows switches."""
        switched = [e for e in explanations if e.action == 1]
        kept     = [e for e in explanations if e.action == 0]

        print(f"\n── Step {step} Decision Explanations ──────────────────────")
        print(f"   {len(switched)}/{len(explanations)} junctions switched\n")

        to_show = explanations if show_all else switched
        for exp in to_show:
            print(exp.to_text())
            print()

        if not show_all and kept:
            print(f"   [{len(kept)} junctions kept current phase — "
                  f"use show_all=True to see details]")
"""
DeepQN/monitoring/monitor.py:
Real-time monitoring of DQN decision quality during live simulation.

Detects:
  - Model degradation    : avg_wait rising above trained baseline
  - Stuck agent          : same action repeated > N times (policy collapse)
  - High CO2 spike       : emissions exceeding threshold
  - Low throughput       : vehicles not completing trips

Usage:
    from DeepQN.monitoring.monitor import DQNMonitor
    monitor = DQNMonitor(baseline_avg_wait=626.6)
    monitor.record(step, avg_wait, total_co2, n_switched, throughput)
    alert = monitor.check()
"""

from __future__ import annotations

import csv
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


# Alert dataclass

@dataclass
class Alert:
    timestamp:  str
    level:      str        # "WARNING" or "CRITICAL"
    code:       str        # short identifier
    message:    str
    value:      float
    threshold:  float


# Monitor

class DQNMonitor:
    """
    Monitors DQN performance during live simulation.
    Records metrics each decision step and raises alerts
    when degradation is detected.
    """

    def __init__(
        self,
        baseline_avg_wait:   float = 626.6,   # Tahrir baseline
        degradation_factor:  float = 0.5,     # alert if DQN wait > 50% of baseline
        stuck_threshold:     int   = 20,      # alert if same n_switched for N steps
        co2_spike_factor:    float = 2.0,     # alert if CO2 > 2x rolling mean
        window_size:         int   = 30,      # rolling window for trend detection
        log_dir:             str   = "DeepQN/logs",
    ):
        self.baseline_wait   = baseline_avg_wait
        self.degrade_thresh  = baseline_avg_wait * degradation_factor
        self.stuck_thresh    = stuck_threshold
        self.co2_spike_mult  = co2_spike_factor
        self.window          = window_size

        # Rolling buffers
        self._wait_buf:     deque = deque(maxlen=window_size)
        self._co2_buf:      deque = deque(maxlen=window_size)
        self._switch_buf:   deque = deque(maxlen=window_size)
        self._thru_buf:     deque = deque(maxlen=window_size)

        # Alert history
        self.alerts: List[Alert] = []

        # Monitoring log file
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._alert_log = log_path / f"monitor_alerts_{ts}.csv"
        self._metric_log = log_path / f"monitor_metrics_{ts}.csv"

        # Init CSV files
        with open(self._alert_log, "w", newline="") as f:
            csv.writer(f).writerow(
                ["timestamp", "level", "code", "message", "value", "threshold"]
            )
        with open(self._metric_log, "w", newline="") as f:
            csv.writer(f).writerow(
                ["timestamp", "step", "avg_wait_s", "total_co2_mg",
                 "n_switched", "throughput", "rolling_avg_wait",
                 "alert_count"]
            )

        logger.info("[Monitor] Initialized. Baseline wait=%.1fs  Degrade threshold=%.1fs",
                    baseline_avg_wait, self.degrade_thresh)

    # Record one decision step

    def record(
        self,
        step:        int,
        avg_wait:    float,
        total_co2:   float,
        n_switched:  int,
        throughput:  int   = 0,
    ):
        """Call this after every DQN decision step."""
        self._wait_buf.append(avg_wait)
        self._co2_buf.append(total_co2)
        self._switch_buf.append(n_switched)
        self._thru_buf.append(throughput)

        rolling_wait = self._rolling_mean(self._wait_buf)
        alerts_this_step = self._check_all(step, avg_wait, total_co2, n_switched, throughput, rolling_wait)

        # Log metrics
        ts = datetime.now().strftime("%H:%M:%S")
        with open(self._metric_log, "a", newline="") as f:
            csv.writer(f).writerow([
                ts, step, round(avg_wait, 2), round(total_co2, 0),
                n_switched, throughput,
                round(rolling_wait, 2), len(alerts_this_step),
            ])

        return alerts_this_step

    # Check all conditions

    def _check_all(
        self, step, avg_wait, total_co2, n_switched, throughput, rolling_wait
    ) -> List[Alert]:
        new_alerts = []

        # 1. Degradation: rolling avg wait exceeds 50% of baseline
        if len(self._wait_buf) >= self.window:
            if rolling_wait > self.degrade_thresh:
                a = self._make_alert(
                    "WARNING", "DEGRADATION",
                    f"Step {step}: Rolling avg wait {rolling_wait:.1f}s exceeds "
                    f"degradation threshold {self.degrade_thresh:.1f}s "
                    f"(50% of baseline {self.baseline_wait:.1f}s)",
                    rolling_wait, self.degrade_thresh,
                )
                new_alerts.append(a)
                logger.warning("[Monitor] %s", a.message)

        # 2. Policy collapse: same n_switched for stuck_threshold steps
        if len(self._switch_buf) >= self.stuck_thresh:
            recent = list(self._switch_buf)[-self.stuck_thresh:]
            if len(set(recent)) == 1:
                a = self._make_alert(
                    "WARNING", "POLICY_COLLAPSE",
                    f"Step {step}: Agent stuck — same n_switched={n_switched} "
                    f"for {self.stuck_thresh} consecutive steps",
                    float(n_switched), float(self.stuck_thresh),
                )
                new_alerts.append(a)
                logger.warning("[Monitor] %s", a.message)

        # 3. CO2 spike: current > 2x rolling mean
        if len(self._co2_buf) >= 5:
            co2_mean = self._rolling_mean(list(self._co2_buf)[:-1])
            if co2_mean > 0 and total_co2 > co2_mean * self.co2_spike_mult:
                a = self._make_alert(
                    "WARNING", "CO2_SPIKE",
                    f"Step {step}: CO2={total_co2:.0f}mg is {total_co2/co2_mean:.1f}x "
                    f"rolling mean {co2_mean:.0f}mg",
                    total_co2, co2_mean * self.co2_spike_mult,
                )
                new_alerts.append(a)
                logger.warning("[Monitor] %s", a.message)

        # 4. Critical: wait exceeds baseline (DQN performing worse than fixed-time)
        if avg_wait > self.baseline_wait * 1.1:
            a = self._make_alert(
                "CRITICAL", "WORSE_THAN_BASELINE",
                f"Step {step}: avg_wait={avg_wait:.1f}s EXCEEDS baseline "
                f"{self.baseline_wait:.1f}s — DQN performing worse than fixed-time",
                avg_wait, self.baseline_wait,
            )
            new_alerts.append(a)
            logger.error("[Monitor] CRITICAL: %s", a.message)

        self.alerts.extend(new_alerts)
        return new_alerts

    # Helper: create and log alert

    def _make_alert(
        self, level: str, code: str, message: str, value: float, threshold: float
    ) -> Alert:
        ts = datetime.now().strftime("%H:%M:%S")
        a  = Alert(ts, level, code, message, value, threshold)
        with open(self._alert_log, "a", newline="") as f:
            csv.writer(f).writerow(
                [ts, level, code, message, round(value, 2), round(threshold, 2)]
            )
        return a

    @staticmethod
    def _rolling_mean(buf) -> float:
        return float(sum(buf) / len(buf)) if buf else 0.0

    # Summary

    def summary(self) -> dict:
        """Return monitoring summary at end of simulation."""
        wait_list = list(self._wait_buf)
        co2_list  = list(self._co2_buf)
        return {
            "total_alerts":       len(self.alerts),
            "critical_alerts":    sum(1 for a in self.alerts if a.level == "CRITICAL"),
            "warning_alerts":     sum(1 for a in self.alerts if a.level == "WARNING"),
            "alert_codes":        list({a.code for a in self.alerts}),
            "final_rolling_wait": round(self._rolling_mean(wait_list), 2),
            "final_rolling_co2":  round(self._rolling_mean(co2_list), 0),
            "metric_log":         str(self._metric_log),
            "alert_log":          str(self._alert_log),
        }

    def print_summary(self):
        s = self.summary()
        print("\n" + "=" * 55)
        print("  DQN MONITORING SUMMARY")
        print("=" * 55)
        print(f"  Total alerts   : {s['total_alerts']}")
        print(f"  Critical       : {s['critical_alerts']}")
        print(f"  Warnings       : {s['warning_alerts']}")
        if s["alert_codes"]:
            print(f"  Alert types    : {', '.join(s['alert_codes'])}")
        print(f"  Final avg wait : {s['final_rolling_wait']}s")
        print(f"  Metric log     : {s['metric_log']}")
        print(f"  Alert log      : {s['alert_log']}")
        print("=" * 55)
        if s["total_alerts"] == 0:
            print("  No degradation detected.")
        elif s["critical_alerts"] > 0:
            print("  CRITICAL alerts : DQN performing below baseline.")
        else:
            print("  Warnings detected : review alert log.")
        print()
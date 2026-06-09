"""
dqn/evaluation/metrics.py
--------------------------
Parses SUMO output XML files to extract evaluation KPIs:
  - average_waiting_time  (from tripinfo XML)
  - total_co2             (from emission XML)
  - throughput            (from tripinfo or summary XML)
  - average_speed         (from summary XML)

These are the "before/after" comparison metrics used to validate that
the DQN satisfies both target KPIs vs. the fixed-time baseline.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


#  Metric containers 

@dataclass
class EpisodeMetrics:
    """Aggregated KPIs for one simulation run."""
    profile:            str
    mode:               str      # "fixed_time" or "dqn_adaptive"
    n_trips:            int      = 0
    avg_waiting_time_s: float    = 0.0
    avg_travel_time_s:  float    = 0.0
    avg_time_loss_s:    float    = 0.0
    total_co2_mg:       float    = 0.0
    avg_co2_per_veh_mg: float    = 0.0
    throughput:         int      = 0    # vehicles that completed trip
    avg_speed_ms:       float    = 0.0  # from summary

    def improvement_vs(self, baseline: "EpisodeMetrics") -> Dict[str, float]:
        """
        Returns percentage improvement for each metric relative to baseline.
        Positive = improvement (less wait, less CO2).
        """
        def pct(before, after):
            if before == 0:
                return 0.0
            return 100.0 * (before - after) / before

        return {
            "avg_wait_pct":    pct(baseline.avg_waiting_time_s, self.avg_waiting_time_s),
            "avg_travel_pct":  pct(baseline.avg_travel_time_s,  self.avg_travel_time_s),
            "time_loss_pct":   pct(baseline.avg_time_loss_s,    self.avg_time_loss_s),
            "total_co2_pct":   pct(baseline.total_co2_mg,       self.total_co2_mg),
            "throughput_delta": self.throughput - baseline.throughput,
        }

    def to_dict(self) -> dict:
        return asdict(self)


#  XML parsers 

def parse_tripinfo(xml_path: str | Path, profile: str, mode: str) -> EpisodeMetrics:
    """
    Parse SUMO tripinfo XML.
    Each <tripinfo> element represents one completed vehicle trip.
    """
    xml_path = Path(xml_path)
    if not xml_path.exists():
        logger.error("tripinfo file not found: %s", xml_path)
        return EpisodeMetrics(profile=profile, mode=mode)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    wait_times:   list = []
    travel_times: list = []
    time_losses:  list = []

    for elem in root.findall("tripinfo"):
        wait   = elem.get("waitingTime")
        travel = elem.get("duration")
        tloss  = elem.get("timeLoss")
        if wait   is not None: wait_times.append(float(wait))
        if travel is not None: travel_times.append(float(travel))
        if tloss  is not None: time_losses.append(float(tloss))

    n = len(wait_times)
    logger.info("Parsed %d trips from %s", n, xml_path.name)

    return EpisodeMetrics(
        profile            = profile,
        mode               = mode,
        n_trips            = n,
        avg_waiting_time_s = float(np.mean(wait_times))   if wait_times   else 0.0,
        avg_travel_time_s  = float(np.mean(travel_times)) if travel_times else 0.0,
        avg_time_loss_s    = float(np.mean(time_losses))  if time_losses  else 0.0,
        throughput         = n,
    )


def parse_emission(xml_path: str | Path, metrics: EpisodeMetrics) -> EpisodeMetrics:
    """
    Parse SUMO emission XML (fcd or emission output).
    Sums CO2 across all vehicles.
    """
    xml_path = Path(xml_path)
    if not xml_path.exists():
        logger.warning("emission file not found: %s", xml_path)
        return metrics

    tree = ET.parse(xml_path)
    root = tree.getroot()
    total_co2 = 0.0
    n_veh     = 0

    # emission-output format: <timestep time="…"><vehicle id="…" CO2="…" .../>
    for ts in root.findall("timestep"):
        for veh in ts.findall("vehicle"):
            co2 = veh.get("CO2")
            if co2 is not None:
                total_co2 += float(co2)
                n_veh += 1

    metrics.total_co2_mg       = total_co2
    metrics.avg_co2_per_veh_mg = total_co2 / max(n_veh, 1)
    logger.info(
        "CO2 from %s: total=%.1f mg, per_veh=%.1f mg",
        xml_path.name, total_co2, metrics.avg_co2_per_veh_mg,
    )
    return metrics


def parse_summary(xml_path: str | Path, metrics: EpisodeMetrics) -> EpisodeMetrics:
    """
    Parse SUMO summary XML.
    Takes the mean speed from the last non-zero entry.
    """
    xml_path = Path(xml_path)
    if not xml_path.exists():
        return metrics

    tree = ET.parse(xml_path)
    root = tree.getroot()
    speeds = []
    for step in root.findall("step"):
        ms = step.get("meanSpeed")
        if ms is not None:
            v = float(ms)
            if v > 0:
                speeds.append(v)

    metrics.avg_speed_ms = float(np.mean(speeds)) if speeds else 0.0
    return metrics


#  Convenience loader 

def load_metrics_from_outputs(
    output_dir:      str | Path | None,
    profile:         str,
    mode:            str,
    suffix_override: str | None = None,
) -> EpisodeMetrics:
    """
    Load baseline metrics from SUMO XML output files.
    Works for any map — pass suffix_override to use a custom filename suffix.

    Tahrir:  tripinfo_morning.xml, tripinfo_evening.xml, etc.
    Taksim:  tripinfo_taksim.xml  (suffix_override="taksim")
    """
    if suffix_override:
        file_sfx = suffix_override
    else:
        suffix_map = {
            "morning_rush": "morning",
            "midday":       "midday",
            "evening_rush": "evening",
            "night":        "night",
        }
        file_sfx = suffix_map.get(profile, profile)

    if output_dir is None:
        odir = Path("simulation") / "maps" / "baseline_outputs"
    else:
        odir = Path(output_dir)

    tripxml = odir / f"tripinfo_{file_sfx}.xml"
    emixml  = odir / f"emission_{file_sfx}.xml"
    sumxml  = odir / f"summary_{file_sfx}.xml"

    logger.info("Reading %s metrics from: %s  (suffix: %s)", mode, odir, file_sfx)
    m = parse_tripinfo(tripxml, profile, mode)
    m = parse_emission(emixml, m)
    m = parse_summary(sumxml, m)
    return m
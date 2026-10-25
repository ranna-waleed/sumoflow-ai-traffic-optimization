"""Strict streaming parsers for comparable SUMO evaluation outputs."""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional

METRIC_DEFINITION = "sumo-xml-reported-arrivals-and-horizon-v3"

class MetricError(ValueError): pass
class IncomparableRunsError(MetricError): pass

@dataclass
class EpisodeMetrics:
    """One run using SUMO's reported-arrival semantics.

    ``successful_arrivals``/``throughput`` count tripinfo records with a
    non-negative arrival time and must match summary ``arrived``.  A non-empty
    ``vaporized`` value is tracked separately because historical SUMO output
    may tag a trip that subsequently has a reported arrival (notably after a
    teleport).  ``removed_or_vaporized`` counts only tagged records without a
    reported arrival; untagged negative-arrival records are unfinished.
    """
    profile: str
    mode: str
    n_trips: Optional[int] = None
    tripinfo_records: Optional[int] = None
    successful_arrivals: Optional[int] = None
    removed_or_vaporized: Optional[int] = None
    teleport_or_vaporization_tagged_arrivals: Optional[int] = None
    unfinished_tripinfo_records: Optional[int] = None
    avg_waiting_time_s: Optional[float] = None
    avg_travel_time_s: Optional[float] = None
    avg_time_loss_s: Optional[float] = None
    total_co2_mg: Optional[float] = None
    network_co2_per_successful_arrival_mg: Optional[float] = None
    mean_successful_trip_co2_mg: Optional[float] = None
    throughput: Optional[int] = None
    summary_arrived: Optional[int] = None
    summary_ended: Optional[int] = None
    running_vehicles: Optional[int] = None
    waiting_vehicles: Optional[int] = None
    unfinished_vehicles: Optional[int] = None
    teleports: Optional[int] = None
    collisions: Optional[int] = None
    avg_speed_ms: Optional[float] = None
    simulation_begin_s: Optional[float] = None
    simulation_end_s: Optional[float] = None
    reached_horizon: Optional[bool] = None
    network_cleared: Optional[bool] = None
    completed: Optional[bool] = None
    valid: bool = True
    errors: list[str] = field(default_factory=list)
    source_files: Dict[str, str] = field(default_factory=dict)
    metric_definition: str = METRIC_DEFINITION

    @property
    def co2_per_completed_trip_mg(self):
        """Compatibility alias: this is network CO2 per successful arrival."""
        return self.network_co2_per_successful_arrival_mg
    @property
    def avg_co2_per_veh_mg(self): return self.network_co2_per_successful_arrival_mg
    def invalidate(self, message):
        self.valid = False
        self.errors.append(message)
    def to_dict(self):
        result = asdict(self)
        result["co2_per_completed_trip_mg"] = self.network_co2_per_successful_arrival_mg
        result["avg_co2_per_veh_mg"] = self.network_co2_per_successful_arrival_mg
        return result
    def improvement_vs(self, baseline) -> Dict[str, object]:
        if not self.valid or not baseline.valid: raise IncomparableRunsError("both runs must be valid")
        if self.profile != baseline.profile: raise IncomparableRunsError("profiles do not match")
        if self.metric_definition != baseline.metric_definition: raise IncomparableRunsError("metric definitions do not match")
        def pct(name, before, after):
            if before is None or after is None or not math.isfinite(before) or not math.isfinite(after):
                raise IncomparableRunsError(f"{name} is missing or non-finite")
            if before == 0:
                return None, f"{name} baseline is zero; percentage is undefined"
            return 100 * (before - after) / before, None
        if self.throughput is None or baseline.throughput is None: raise IncomparableRunsError("throughput is missing")
        values = {}
        for key, label, before, after in (
            ("avg_wait", "average waiting time", baseline.avg_waiting_time_s, self.avg_waiting_time_s),
            ("avg_travel", "average travel time", baseline.avg_travel_time_s, self.avg_travel_time_s),
            ("time_loss", "average time loss", baseline.avg_time_loss_s, self.avg_time_loss_s),
            ("total_co2", "total CO2", baseline.total_co2_mg, self.total_co2_mg)):
            percentage, reason = pct(label, before, after)
            values[f"{key}_pct"] = percentage
            values[f"{key}_pct_unavailable_reason"] = reason
            values[f"{key}_absolute_delta"] = after - before
        values["throughput_delta"] = self.throughput - baseline.throughput
        return values

def _number(raw, name, path, nonnegative=True):
    try: value = float(raw)
    except (TypeError, ValueError) as exc: raise MetricError(f"invalid '{name}' value in {path}: {raw!r}") from exc
    if not math.isfinite(value) or (nonnegative and value < 0): raise MetricError(f"invalid '{name}' value in {path}: {raw!r}")
    return value

def _required(elem, name, path, nonnegative=True):
    if elem.get(name) is None: raise MetricError(f"missing '{name}' attribute in {path}")
    return _number(elem.get(name), name, path, nonnegative)

def _count(attrs, name, path, default=None):
    raw = attrs.get(name)
    if raw is None:
        if default is not None: return default
        raise MetricError(f"missing '{name}' attribute in {path}")
    try: value = int(raw)
    except ValueError as exc: raise MetricError(f"invalid '{name}' count in {path}: {raw!r}") from exc
    if value < 0: raise MetricError(f"invalid negative '{name}' count in {path}")
    return value

def parse_tripinfo(xml_path, profile, mode):
    path = Path(xml_path); m = EpisodeMetrics(profile, mode); m.source_files["tripinfo"] = str(path.resolve())
    if not path.is_file(): m.invalidate(f"tripinfo file not found: {path}"); return m
    records = arrivals = removed = unfinished = tagged_arrivals = 0
    wait = duration = loss = trip_co2 = 0.0; trip_co2_count = 0
    try:
        with path.open("rb") as stream:
            for _, elem in ET.iterparse(stream, events=("end",)):
                if elem.tag != "tripinfo": continue
                records += 1; arrival = _required(elem, "arrival", path, False)
                vap = elem.get("vaporized")
                tagged = vap is not None and vap.strip().lower() not in {"", "0", "false", "none"}
                if arrival >= 0:
                    arrivals += 1
                    if tagged: tagged_arrivals += 1
                    wait += _required(elem, "waitingTime", path); duration += _required(elem, "duration", path); loss += _required(elem, "timeLoss", path)
                    emissions = elem.find("emissions")
                    if emissions is not None and emissions.get("CO2_abs") is not None:
                        trip_co2 += _required(emissions, "CO2_abs", path); trip_co2_count += 1
                elif tagged: removed += 1
                else: unfinished += 1
                elem.clear()
    except (ET.ParseError, MetricError, OSError) as exc:
        prefix = "malformed XML" if isinstance(exc, ET.ParseError) else "tripinfo parse failed"
        m.invalidate(f"{prefix} in {path}: {exc}"); return m
    m.n_trips = m.tripinfo_records = records; m.successful_arrivals = m.throughput = arrivals
    m.removed_or_vaporized = removed; m.unfinished_tripinfo_records = unfinished
    m.teleport_or_vaporization_tagged_arrivals = tagged_arrivals
    if arrivals:
        m.avg_waiting_time_s = wait / arrivals; m.avg_travel_time_s = duration / arrivals; m.avg_time_loss_s = loss / arrivals
        if trip_co2_count == arrivals: m.mean_successful_trip_co2_mg = trip_co2 / arrivals
    return m

def parse_emission(xml_path, metrics, step_length_s=1.0, emission_step_scaled=False):
    path = Path(xml_path); metrics.source_files["emission"] = str(path.resolve())
    if not path.is_file(): metrics.invalidate(f"emission file not found: {path}"); return metrics
    if not math.isfinite(step_length_s) or step_length_s <= 0: metrics.invalidate("emission step length must be finite and positive"); return metrics
    total = 0.0
    try:
        with path.open("rb") as stream:
            for _, elem in ET.iterparse(stream, events=("end",)):
                if elem.tag == "vehicle":
                    value = _required(elem, "CO2", path); total += value if emission_step_scaled else value * step_length_s
                elem.clear()
        if not math.isfinite(total): raise MetricError("non-finite total CO2")
    except (ET.ParseError, MetricError, OSError) as exc: metrics.invalidate(f"emission parse failed in {path}: {exc}"); return metrics
    metrics.total_co2_mg = total
    metrics.network_co2_per_successful_arrival_mg = total / metrics.successful_arrivals if metrics.successful_arrivals else None
    return metrics

def parse_summary(xml_path, metrics, *, expected_begin_s, expected_end_s, step_length_s):
    path = Path(xml_path); metrics.source_files["summary"] = str(path.resolve())
    if not path.is_file(): metrics.invalidate(f"summary file not found: {path}"); return metrics
    if any(not math.isfinite(x) for x in (expected_begin_s, expected_end_s, step_length_s)) or step_length_s <= 0 or expected_end_s <= expected_begin_s:
        metrics.invalidate("invalid expected simulation horizon or step length"); return metrics
    first = None; last = None; speed = 0.0; speed_n = 0
    try:
        with path.open("rb") as stream:
            for _, elem in ET.iterparse(stream, events=("end",)):
                if elem.tag == "step":
                    t = _required(elem, "time", path); first = t if first is None else first; last = dict(elem.attrib)
                    if elem.get("meanSpeed") is not None:
                        raw_speed = _number(elem.get("meanSpeed"), "meanSpeed", path, nonnegative=False)
                        if raw_speed == -1.0:  # SUMO sentinel when no vehicle speed is available
                            pass
                        elif raw_speed < 0:
                            raise MetricError(f"invalid negative 'meanSpeed' value in {path}: {raw_speed}")
                        else:
                            speed += raw_speed; speed_n += 1
                elem.clear()
        if last is None: raise MetricError("summary contains no step elements")
        end = _number(last.get("time"), "time", path)
        metrics.simulation_begin_s = first; metrics.simulation_end_s = end
        metrics.summary_arrived = _count(last, "arrived", path); metrics.summary_ended = _count(last, "ended", path)
        metrics.running_vehicles = _count(last, "running", path); metrics.waiting_vehicles = _count(last, "waiting", path)
        metrics.unfinished_vehicles = metrics.running_vehicles + metrics.waiting_vehicles
        metrics.teleports = _count(last, "teleports", path, 0); metrics.collisions = _count(last, "collisions", path, 0)
        metrics.avg_speed_ms = speed / speed_n if speed_n else None
        metrics.reached_horizon = end + step_length_s >= expected_end_s - 1e-9
        metrics.completed = metrics.reached_horizon; metrics.network_cleared = metrics.unfinished_vehicles == 0
        if first is None or abs(first - expected_begin_s) > step_length_s + 1e-9:
            raise MetricError(f"summary begins at {first}, expected {expected_begin_s}")
        if metrics.summary_ended < metrics.summary_arrived: raise MetricError("summary ended is smaller than arrived")
        if metrics.successful_arrivals != metrics.summary_arrived: raise MetricError(f"tripinfo arrivals {metrics.successful_arrivals} != summary arrived {metrics.summary_arrived}")
        summary_removed = metrics.summary_ended - metrics.summary_arrived
        if metrics.removed_or_vaporized != summary_removed:
            raise MetricError(f"tripinfo removed/vaporized {metrics.removed_or_vaporized} != summary non-arrival endings {summary_removed}")
        if not metrics.reached_horizon: raise MetricError(f"simulation ended at {end}, before horizon {expected_end_s}")
    except (ET.ParseError, MetricError, OSError) as exc: metrics.invalidate(f"summary parse failed in {path}: {exc}")
    return metrics

def load_metrics_from_paths(*, tripinfo_path, emission_path, summary_path, profile, mode, step_length_s=1.0,
                            expected_begin_s=0.0, expected_end_s=None, emission_step_scaled=False):
    m = parse_tripinfo(tripinfo_path, profile, mode)
    m = parse_emission(emission_path, m, step_length_s, emission_step_scaled)
    if expected_end_s is None: m.invalidate("expected simulation horizon is required"); return m
    return parse_summary(summary_path, m, expected_begin_s=expected_begin_s, expected_end_s=expected_end_s, step_length_s=step_length_s)

def load_metrics_from_outputs(output_dir, profile, mode, suffix_override=None, step_length_s=1.0,
                              expected_begin_s=0.0, expected_end_s=None, emission_step_scaled=False):
    suffix = suffix_override or {"morning_rush":"morning","midday":"midday","evening_rush":"evening","night":"night"}.get(profile, profile)
    directory = Path(output_dir or "simulation/maps/baseline_outputs")
    return load_metrics_from_paths(tripinfo_path=directory/f"tripinfo_{suffix}.xml", emission_path=directory/f"emission_{suffix}.xml",
        summary_path=directory/f"summary_{suffix}.xml", profile=profile, mode=mode, step_length_s=step_length_s,
        expected_begin_s=expected_begin_s, expected_end_s=expected_end_s, emission_step_scaled=emission_step_scaled)

"""Strict streaming parsers for comparable SUMO evaluation outputs."""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional

METRIC_DEFINITION = "sumo-xml-demand-normalized-horizon-wait-v5"

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
    scheduled_demand: Optional[int] = None
    inserted_vehicles: Optional[int] = None
    undeparted_vehicles: Optional[int] = None
    demand_normalized_waiting_s: Optional[float] = None
    censored_preinsertion_delay_per_due_vehicle_s: Optional[float] = None
    mean_departure_delay_inserted_s: Optional[float] = None
    insertion_rate: Optional[float] = None
    completion_rate: Optional[float] = None
    conditional_completion_rate: Optional[float] = None
    arrival_throughput_veh_per_hour: Optional[float] = None
    avg_waiting_time_s: Optional[float] = None
    avg_travel_time_s: Optional[float] = None
    avg_time_loss_s: Optional[float] = None
    total_co2_mg: Optional[float] = None
    network_co2_per_successful_arrival_mg: Optional[float] = None
    mean_successful_trip_co2_mg: Optional[float] = None
    throughput: Optional[int] = None
    summary_arrived: Optional[int] = None
    summary_ended: Optional[int] = None
    summary_inserted: Optional[int] = None
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
    _trip_records: Dict[str, dict] = field(default_factory=dict, repr=False, compare=False)

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
        result.pop("_trip_records", None)
        result["co2_per_completed_trip_mg"] = self.network_co2_per_successful_arrival_mg
        result["avg_co2_per_veh_mg"] = self.network_co2_per_successful_arrival_mg
        return result
    def improvement_vs(self, baseline) -> Dict[str, object]:
        if not self.valid or not baseline.valid: raise IncomparableRunsError("both runs must be valid")
        if self.profile != baseline.profile: raise IncomparableRunsError("profiles do not match")
        if self.metric_definition != baseline.metric_definition: raise IncomparableRunsError("metric definitions do not match")
        def compare(name, before, after):
            if before is None or after is None:
                return None, None, f"{name} is unavailable for one or both runs"
            if not math.isfinite(before) or not math.isfinite(after):
                raise IncomparableRunsError(f"{name} is non-finite")
            delta = after - before
            if before == 0:
                return delta, None, f"{name} baseline is zero; percentage is undefined"
            return delta, 100 * (before - after) / before, None
        primary_before = baseline.demand_normalized_waiting_s
        primary_after = self.demand_normalized_waiting_s
        if primary_before is None or primary_after is None:
            raise IncomparableRunsError("required demand-normalized horizon waiting is missing")
        primary_delta, primary_pct, primary_reason = compare(
            "demand-normalized horizon waiting", primary_before, primary_after
        )
        if self.throughput is None or baseline.throughput is None: raise IncomparableRunsError("throughput is missing")
        values = {
            "demand_normalized_waiting_absolute_delta": primary_delta,
            "demand_normalized_waiting_pct": primary_pct,
            "demand_normalized_waiting_pct_unavailable_reason": primary_reason,
        }
        for key, label, before, after in (
            ("avg_wait", "average waiting time", baseline.avg_waiting_time_s, self.avg_waiting_time_s),
            ("avg_travel", "average travel time", baseline.avg_travel_time_s, self.avg_travel_time_s),
            ("time_loss", "average time loss", baseline.avg_time_loss_s, self.avg_time_loss_s),
            ("total_co2", "total CO2", baseline.total_co2_mg, self.total_co2_mg)):
            delta, percentage, reason = compare(label, before, after)
            values[f"{key}_pct"] = percentage
            values[f"{key}_pct_unavailable_reason"] = reason
            values[f"{key}_absolute_delta"] = delta
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

def parse_scheduled_demand(route_paths: Iterable[str | Path], begin_s: float, end_s: float) -> Dict[str, float]:
    """Return explicit vehicle IDs eligible in SUMO's half-open [begin, end) interval.

    Research evaluation deliberately rejects flows, trips, non-numeric departure
    procedures, and duplicate IDs because they do not provide an unambiguous
    frozen per-vehicle denominator.
    """
    if not math.isfinite(begin_s) or not math.isfinite(end_s) or end_s <= begin_s:
        raise MetricError("demand interval must be finite with end greater than begin")
    due: Dict[str, float] = {}
    seen: set[str] = set()
    paths = [Path(path) for path in route_paths]
    if not paths:
        raise MetricError("at least one route file is required for the demand contract")
    for path in paths:
        if not path.is_file():
            raise MetricError(f"route file not found: {path}")
        try:
            with path.open("rb") as stream:
                for _, elem in ET.iterparse(stream, events=("end",)):
                    if elem.tag in {"flow", "trip"}:
                        raise MetricError(
                            f"unsupported generated demand element <{elem.tag}> in {path}; "
                            "freeze it to explicit vehicles first"
                        )
                    if elem.tag == "vehicle":
                        vehicle_id = elem.get("id")
                        if not vehicle_id:
                            raise MetricError(f"vehicle without id in {path}")
                        if vehicle_id in seen:
                            raise MetricError(f"duplicate demand vehicle id {vehicle_id!r}")
                        seen.add(vehicle_id)
                        depart = _required(elem, "depart", path)
                        if begin_s <= depart < end_s:
                            due[vehicle_id] = depart
                    elem.clear()
        except (ET.ParseError, OSError) as exc:
            raise MetricError(f"demand parse failed in {path}: {exc}") from exc
    return due

def parse_tripinfo(xml_path, profile, mode):
    path = Path(xml_path); m = EpisodeMetrics(profile, mode); m.source_files["tripinfo"] = str(path.resolve())
    if not path.is_file(): m.invalidate(f"tripinfo file not found: {path}"); return m
    records = arrivals = removed = unfinished = tagged_arrivals = 0
    wait = duration = loss = trip_co2 = 0.0; trip_co2_count = 0
    try:
        with path.open("rb") as stream:
            for _, elem in ET.iterparse(stream, events=("end",)):
                if elem.tag != "tripinfo": continue
                records += 1
                vehicle_id = elem.get("id") or f"__anonymous_record_{records}"
                if vehicle_id in m._trip_records:
                    raise MetricError(f"duplicate tripinfo id in {path}: {vehicle_id!r}")
                arrival = _required(elem, "arrival", path, False)
                depart = (
                    _number(elem.get("depart"), "depart", path, False)
                    if elem.get("depart") is not None else None
                )
                waiting_time = _required(elem, "waitingTime", path)
                reported_depart_delay = (
                    _number(elem.get("departDelay"), "departDelay", path)
                    if elem.get("departDelay") is not None else None
                )
                vap = elem.get("vaporized")
                tagged = vap is not None and vap.strip().lower() not in {"", "0", "false", "none"}
                vaporized_reason = vap.strip().lower() if tagged else None
                m._trip_records[vehicle_id] = {
                    "arrival": arrival, "depart": depart,
                    "waiting_time": waiting_time, "tagged": tagged,
                    "vaporized_reason": vaporized_reason,
                    "depart_delay": reported_depart_delay,
                }
                if arrival >= 0:
                    arrivals += 1
                    if tagged: tagged_arrivals += 1
                    wait += waiting_time; duration += _required(elem, "duration", path); loss += _required(elem, "timeLoss", path)
                    emissions = elem.find("emissions")
                    if emissions is not None and emissions.get("CO2_abs") is not None:
                        trip_co2 += _required(emissions, "CO2_abs", path); trip_co2_count += 1
                elif tagged and vaporized_reason != "end":
                    removed += 1
                else:
                    # SUMO 1.26 writes vaporized="end" for both running and
                    # undeparted records emitted at a fixed horizon. It is a
                    # horizon marker, not a non-arrival removal.
                    unfinished += 1
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

def reconcile_demand(metrics: EpisodeMetrics, route_paths: Iterable[str | Path], begin_s: float, end_s: float) -> EpisodeMetrics:
    """Validate all due demand IDs and calculate horizon-censored outcomes."""
    try:
        due = parse_scheduled_demand(route_paths, begin_s, end_s)
        all_demand = parse_scheduled_demand(route_paths, -1e300, 1e300)
        due_ids = set(due)
        record_ids = set(metrics._trip_records)
        # SUMO 1.26 may emit write-undeparted records scheduled exactly at
        # --end even though the end instant is not simulated. Validate and
        # exclude those boundary markers from the eligible denominator.
        # Parsed explicit departure timestamps use the same numeric domain as
        # the configured horizon. Exact equality keeps this set disjoint from
        # eligible [begin, end) demand; relative tolerance would incorrectly
        # classify near-end departures at large simulation times.
        boundary_ids = {vehicle_id for vehicle_id, depart in all_demand.items()
                        if depart == end_s and vehicle_id not in due_ids}
        for vehicle_id in record_ids & boundary_ids:
            record = metrics._trip_records[vehicle_id]
            if record["depart"] >= 0 or record["arrival"] >= 0 or record["vaporized_reason"] not in {None, "end"}:
                raise MetricError(f"end-boundary vehicle {vehicle_id!r} has unexpected execution state")
        unknown = sorted(record_ids - due_ids - boundary_ids)
        missing = sorted(due_ids - record_ids)
        if unknown or missing:
            raise MetricError(
                f"tripinfo/demand ID mismatch; unknown={unknown[:5]}, missing={missing[:5]}"
            )
        inserted = undeparted = 0
        pre_delay_total = inserted_delay_total = on_network_wait_total = 0.0
        for vehicle_id, intended in due.items():
            record = metrics._trip_records[vehicle_id]
            depart = record["depart"]
            if depart is None:
                raise MetricError(f"tripinfo record {vehicle_id!r} has no depart time")
            if depart >= 0:
                if depart + 1e-9 < intended:
                    raise MetricError(f"vehicle {vehicle_id!r} departed before intended time")
                inserted += 1
                delay = depart - intended
                if record["depart_delay"] is not None and not math.isclose(
                    record["depart_delay"], delay, abs_tol=1e-6
                ):
                    raise MetricError(f"vehicle {vehicle_id!r} has contradictory departDelay")
                inserted_delay_total += delay
                pre_delay_total += delay
                on_network_wait_total += record["waiting_time"]
            else:
                if record["arrival"] >= 0 or record["vaporized_reason"] not in {None, "end"}:
                    raise MetricError(f"undeparted vehicle {vehicle_id!r} has arrival/removal state")
                undeparted += 1
                delay = end_s - intended
                if record["depart_delay"] is not None and not math.isclose(
                    record["depart_delay"], delay, abs_tol=1e-6
                ):
                    raise MetricError(f"vehicle {vehicle_id!r} has contradictory censored departDelay")
                pre_delay_total += delay
        due_count = len(due)
        metrics.scheduled_demand = due_count
        metrics.inserted_vehicles = inserted
        metrics.undeparted_vehicles = undeparted
        if due_count:
            metrics.demand_normalized_waiting_s = (
                pre_delay_total + on_network_wait_total
            ) / due_count
            metrics.censored_preinsertion_delay_per_due_vehicle_s = pre_delay_total / due_count
            metrics.insertion_rate = inserted / due_count
            metrics.completion_rate = (metrics.successful_arrivals or 0) / due_count
        else:
            metrics.demand_normalized_waiting_s = None
            metrics.censored_preinsertion_delay_per_due_vehicle_s = None
            metrics.insertion_rate = None
            metrics.completion_rate = None
        metrics.mean_departure_delay_inserted_s = (
            inserted_delay_total / inserted if inserted else None
        )
        metrics.conditional_completion_rate = (
            (metrics.successful_arrivals or 0) / inserted if inserted else None
        )
        duration = end_s - begin_s
        metrics.arrival_throughput_veh_per_hour = (
            (metrics.successful_arrivals or 0) * 3600.0 / duration if duration > 0 else None
        )
    except MetricError as exc:
        metrics.invalidate(f"demand reconciliation failed: {exc}")
    return metrics

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
        metrics.summary_inserted = _count(last, "inserted", path, None) if last.get("inserted") is not None else None
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
                            expected_begin_s=0.0, expected_end_s=None, emission_step_scaled=False,
                            route_paths=(), require_demand_contract=False):
    m = parse_tripinfo(tripinfo_path, profile, mode)
    m = parse_emission(emission_path, m, step_length_s, emission_step_scaled)
    if expected_end_s is None: m.invalidate("expected simulation horizon is required"); return m
    m = parse_summary(summary_path, m, expected_begin_s=expected_begin_s, expected_end_s=expected_end_s, step_length_s=step_length_s)
    if route_paths:
        m = reconcile_demand(m, route_paths, expected_begin_s, expected_end_s)
        if m.valid:
            if m.inserted_vehicles != (m.summary_arrived + m.running_vehicles + m.removed_or_vaporized):
                m.invalidate("inserted demand does not reconcile with arrived + running + removed")
            if m.undeparted_vehicles != m.waiting_vehicles:
                m.invalidate("undeparted demand does not reconcile with summary waiting")
            if m.summary_inserted is None:
                m.invalidate("summary inserted count is required for demand reconciliation")
            elif m.inserted_vehicles != m.summary_inserted:
                m.invalidate("inserted demand does not reconcile with summary inserted")
    elif require_demand_contract:
        m.invalidate("route files are required for the demand-normalized metric contract")
    return m

def load_metrics_from_outputs(output_dir, profile, mode, suffix_override=None, step_length_s=1.0,
                              expected_begin_s=0.0, expected_end_s=None, emission_step_scaled=False,
                              route_paths=(), require_demand_contract=False):
    suffix = suffix_override or {"morning_rush":"morning","midday":"midday","evening_rush":"evening","night":"night"}.get(profile, profile)
    directory = Path(output_dir or "simulation/maps/baseline_outputs")
    return load_metrics_from_paths(tripinfo_path=directory/f"tripinfo_{suffix}.xml", emission_path=directory/f"emission_{suffix}.xml",
        summary_path=directory/f"summary_{suffix}.xml", profile=profile, mode=mode, step_length_s=step_length_s,
        expected_begin_s=expected_begin_s, expected_end_s=expected_end_s, emission_step_scaled=emission_step_scaled,
        route_paths=route_paths, require_demand_contract=require_demand_contract)

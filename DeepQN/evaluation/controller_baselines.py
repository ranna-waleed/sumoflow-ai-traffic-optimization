"""Fail-closed definitions for non-learning traffic-signal baselines.

The classes in this module validate controller inputs; they do not infer missing
movements or invent a traffic-light program.  Publication runs must therefore
provide a reviewed definition whose hash is recorded in the run manifest.
"""
from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


CONTROLLER_CONTRACT_VERSION = "sumoflow-controller-baselines-v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SignalProgram:
    tls_id: str
    program_id: str
    program_type: str
    phases: tuple[dict, ...]
    source_path: str
    source_sha256: str
    program_sha256: str
    parameters: tuple[tuple[str, str], ...] = ()


def _program_payload(tls_id: str, program_id: str, program_type: str,
                     phases: tuple[dict, ...], parameters=()) -> dict:
    return {"tls_id": tls_id, "program_id": program_id, "program_type": program_type,
            "phases": list(phases), "parameters": dict(parameters)}


def _program_hash(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_signal_program(path: str | Path, tls_id: str, expected_type: str,
                        network_path: str | Path | None = None) -> SignalProgram:
    """Load one explicit program and verify its timing/state contract."""
    source = Path(path).resolve()
    matches = [node for node in ET.parse(source).getroot().iter("tlLogic")
               if node.get("id") == tls_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one tlLogic for {tls_id!r}; found {len(matches)}")
    node = matches[0]
    actual_type = node.get("type")
    if actual_type != expected_type:
        raise ValueError(f"TLS {tls_id!r} is {actual_type!r}, not {expected_type!r}")
    phases = []
    state_length = None
    for index, phase in enumerate(node.findall("phase")):
        state = phase.get("state", "")
        if not state or (state_length is not None and len(state) != state_length):
            raise ValueError(f"TLS {tls_id!r} phase {index} has an invalid state")
        state_length = len(state)
        duration = float(phase.get("duration", "nan"))
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"TLS {tls_id!r} phase {index} has invalid duration")
        record = {"index": index, "state": state, "duration_s": duration}
        if expected_type == "actuated" and any(key in phase.attrib for key in ("minDur", "maxDur")):
            minimum = float(phase.get("minDur", duration))
            maximum = float(phase.get("maxDur", duration))
            if not all(math.isfinite(value) and value > 0 for value in (minimum, maximum)) or minimum > maximum:
                raise ValueError(f"TLS {tls_id!r} phase {index} has invalid actuated limits")
            record.update(min_duration_s=minimum, max_duration_s=maximum)
        phases.append(record)
    if not phases:
        raise ValueError(f"TLS {tls_id!r} contains no phases")
    if network_path is not None:
        network = ET.parse(Path(network_path)).getroot()
        indices = [int(conn.get("linkIndex")) for conn in network.iter("connection")
                   if conn.get("tl") == tls_id and conn.get("linkIndex") is not None]
        if not indices:
            raise ValueError(f"network has no controlled links for TLS {tls_id!r}")
        required_state_length = max(indices) + 1
        if state_length != required_state_length:
            raise ValueError(
                f"TLS {tls_id!r} state width {state_length} does not match "
                f"network controlled-link width {required_state_length}"
            )
    params = tuple(sorted((p.get("key"), p.get("value")) for p in node.findall("param")))
    if expected_type == "actuated":
        variable = [phase for phase in phases if "min_duration_s" in phase]
        if not variable or not {"max-gap", "detector-gap", "passing-time"}.issubset(dict(params)):
            raise ValueError("actuated program requires variable phases and explicit gap parameters")
    phase_tuple = tuple(phases)
    payload = _program_payload(tls_id, node.get("programID", ""), actual_type,
                               phase_tuple, params)
    return SignalProgram(tls_id, node.get("programID", ""), actual_type,
                         phase_tuple, str(source), _sha256(source),
                         _program_hash(payload), params)


def verify_loaded_signal_programs(sumocfg_path: str | Path,
                                  programs: list[SignalProgram],
                                  sumo_binary: str = "sumo") -> list[dict]:
    """Start the configured SUMO inputs and compare the selected live TLS programs."""
    import traci
    import traci.constants as tc
    expected_types = {"static": tc.TRAFFICLIGHT_TYPE_STATIC,
                      "actuated": tc.TRAFFICLIGHT_TYPE_ACTUATED}
    label = f"baseline-preflight-{uuid.uuid4().hex}"
    connection = None
    try:
        traci.start([sumo_binary, "-c", str(Path(sumocfg_path).resolve()),
                     "--no-step-log", "true"], label=label)
        connection = traci.getConnection(label)
        verified = []
        for expected in programs:
            selected_id = connection.trafficlight.getProgram(expected.tls_id)
            if selected_id != expected.program_id:
                raise RuntimeError(
                    f"SUMO selected TLS program {selected_id!r} for {expected.tls_id}, "
                    f"not declared {expected.program_id!r}"
                )
            matches = [logic for logic in connection.trafficlight.getAllProgramLogics(expected.tls_id)
                       if logic.programID == expected.program_id]
            if len(matches) != 1:
                raise RuntimeError(f"SUMO did not load exactly one declared program for {expected.tls_id}")
            logic = matches[0]
            if logic.type != expected_types[expected.program_type]:
                raise RuntimeError(f"SUMO program type mismatch for {expected.tls_id}")
            observed_phases = []
            for index, phase in enumerate(logic.phases):
                item = {"index": index, "state": phase.state,
                        "duration_s": float(phase.duration)}
                if expected.program_type == "actuated" and (
                        phase.minDur != phase.duration or phase.maxDur != phase.duration):
                    item.update(min_duration_s=float(phase.minDur),
                                max_duration_s=float(phase.maxDur))
                observed_phases.append(item)
            observed_parameters = tuple(sorted(logic.getParameters().items()))
            payload = _program_payload(expected.tls_id, logic.programID,
                                       expected.program_type, tuple(observed_phases),
                                       observed_parameters)
            observed_hash = _program_hash(payload)
            if observed_hash != expected.program_sha256:
                raise RuntimeError(
                    f"SUMO loaded TLS program definition differs for {expected.tls_id}: "
                    f"declared={expected.program_sha256} observed={observed_hash}"
                )
            verified.append({"tls_id": expected.tls_id, "program_id": logic.programID,
                             "program_type": expected.program_type,
                             "observed_program_sha256": observed_hash})
        return verified
    finally:
        if connection is not None:
            connection.close(True)


def _verify_loaded_signal_programs_worker(pipe, sumocfg_path: str,
                                           programs: list[SignalProgram],
                                           sumo_binary: str) -> None:
    """Isolated worker so a stuck SUMO/TraCI preflight can be terminated."""
    try:
        pipe.send((True, verify_loaded_signal_programs(
            sumocfg_path, programs, sumo_binary)))
    except BaseException as exc:  # propagate a stable diagnostic across processes
        pipe.send((False, f"{type(exc).__name__}: {exc}"))
    finally:
        pipe.close()


def verify_loaded_signal_programs_bounded(
        sumocfg_path: str | Path, programs: list[SignalProgram],
        sumo_binary: str = "sumo", *, wall_clock_timeout_s: float) -> list[dict]:
    """Verify live programs in a killable process with an explicit wall limit."""
    timeout = float(wall_clock_timeout_s)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("live-program preflight timeout must be finite and positive")
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_verify_loaded_signal_programs_worker,
        args=(child, str(Path(sumocfg_path).resolve()), programs, sumo_binary),
        name="sumoflow-live-program-preflight",
    )
    process.start()
    child.close()
    try:
        if not parent.poll(timeout):
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
            raise TimeoutError(
                f"live SUMO program preflight exceeded wall-clock timeout {timeout:g}s"
            )
        success, payload = parent.recv()
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            raise RuntimeError(
                "live SUMO program preflight worker did not shut down cleanly"
            )
        if not success:
            raise RuntimeError(f"live SUMO program preflight failed: {payload}")
        if process.exitcode != 0:
            raise RuntimeError(
                f"live SUMO program preflight worker exited {process.exitcode}"
            )
        return payload
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)


@dataclass(frozen=True)
class Movement:
    upstream_lane: str
    downstream_lane: str
    link_index: int
    phase_indices: tuple[int, ...]
    saturation_weight: float


@dataclass(frozen=True)
class MaxPressureDefinition:
    tls_id: str
    movements: tuple[Movement, ...]
    definition_path: str
    definition_sha256: str
    network_sha256: str


def load_max_pressure_definition(path: str | Path, network_path: str | Path,
                                 tls_id: str) -> MaxPressureDefinition:
    """Validate movements against physical controlled links and served phases."""
    source, network = Path(path).resolve(), Path(network_path).resolve()
    with source.open(encoding="utf-8") as stream:
        raw = json.load(stream)
    if raw.get("schema") != "sumoflow-max-pressure-v2" or raw.get("tls_id") != tls_id:
        raise ValueError("max-pressure schema or TLS identity mismatch")
    root = ET.parse(network).getroot()
    logic = next((x for x in root.iter("tlLogic") if x.get("id") == tls_id), None)
    if logic is None:
        raise ValueError(f"network has no TLS {tls_id!r}")
    phase_states = [p.get("state", "") for p in logic.findall("phase")]
    connections = {}
    for conn in root.iter("connection"):
        if conn.get("tl") != tls_id or conn.get("linkIndex") is None:
            continue
        key = (f"{conn.get('from')}_{conn.get('fromLane')}",
               f"{conn.get('to')}_{conn.get('toLane')}", int(conn.get("linkIndex")))
        connections[key] = True
    movements = []
    seen_links = set()
    for item in raw.get("movements", []):
        service_phases = tuple(sorted({int(value) for value in item["phase_indices"]}))
        movement = Movement(item["upstream_lane"], item["downstream_lane"],
                            int(item["link_index"]), service_phases,
                            float(item["saturation_weight"]))
        key = (movement.upstream_lane, movement.downstream_lane, movement.link_index)
        if key not in connections:
            raise ValueError(f"movement is not a controlled network connection: {key}")
        if movement.link_index in seen_links:
            raise ValueError(f"duplicate controlled link {movement.link_index}")
        if not math.isfinite(movement.saturation_weight) or movement.saturation_weight <= 0:
            raise ValueError("saturation weights must be finite and positive")
        if not movement.phase_indices or any(
                phase < 0 or phase >= len(phase_states) for phase in movement.phase_indices):
            raise ValueError("movement phase index is outside the TLS program")
        actual_service = tuple(index for index, state in enumerate(phase_states)
                               if movement.link_index < len(state)
                               and state[movement.link_index] in "Gg")
        if movement.phase_indices != actual_service:
            raise ValueError(
                f"movement service inventory mismatch for link {movement.link_index}: "
                f"declared={movement.phase_indices} physical={actual_service}"
            )
        seen_links.add(movement.link_index)
        movements.append(movement)
    if not movements:
        raise ValueError("max-pressure definition contains no movements")
    required_links = {key[2] for key in connections}
    if seen_links != required_links:
        raise ValueError(
            "max-pressure definition must account for every controlled link; "
            f"missing={sorted(required_links - seen_links)} extra={sorted(seen_links - required_links)}"
        )
    return MaxPressureDefinition(tls_id, tuple(movements), str(source), _sha256(source), _sha256(network))


class MaxPressureController:
    """Select the phase maximizing weighted upstream-minus-downstream queue."""

    def __init__(self, definition: MaxPressureDefinition):
        self.definition = definition

    def pressures(self, queue_by_lane: Mapping[str, float]) -> dict[int, float]:
        result: dict[int, float] = {}
        required = {m.upstream_lane for m in self.definition.movements} | {
            m.downstream_lane for m in self.definition.movements}
        missing = sorted(required - set(queue_by_lane))
        if missing:
            raise ValueError(f"missing queue measurements: {missing}")
        for lane in required:
            value = float(queue_by_lane[lane])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid queue for lane {lane!r}")
        for movement in self.definition.movements:
            pressure = movement.saturation_weight * (
                float(queue_by_lane[movement.upstream_lane]) -
                float(queue_by_lane[movement.downstream_lane]))
            for phase_index in movement.phase_indices:
                result[phase_index] = result.get(phase_index, 0.0) + pressure
        return result

    def select_phase(self, queue_by_lane: Mapping[str, float]) -> int:
        values = self.pressures(queue_by_lane)
        return min(values, key=lambda phase: (-values[phase], phase))


class MaxPressureActionProvider:
    """Adapt validated pressure controllers to the environment's keep/switch API."""

    def __init__(self, controllers: Mapping[str, MaxPressureController]):
        if not controllers:
            raise ValueError("at least one max-pressure controller is required")
        self.controllers = dict(controllers)
        self.tls_ids = tuple(sorted(self.controllers))

    def actions(self, env, _observations) -> dict[str, int]:
        # Read and validate every lane before producing any action, so a failed
        # sensor query cannot partially mutate phase-controller state.
        import traci
        selected = {}
        for tls_id in self.tls_ids:
            controller = self.controllers[tls_id]
            lanes = {m.upstream_lane for m in controller.definition.movements} | {
                m.downstream_lane for m in controller.definition.movements}
            queues = {lane: float(traci.lane.getLastStepHaltingNumber(lane))
                      for lane in sorted(lanes)}
            selected[tls_id] = controller.select_phase(queues)
        controller_states = env.phase_controller_states()
        if set(controller_states) != set(self.tls_ids):
            raise ValueError("max-pressure TLS set does not match the environment")
        actions = {}
        for tls_id in self.tls_ids:
            state = controller_states[tls_id]
            eligible = tuple(sorted(self.controllers[tls_id].pressures(
                {lane: 0.0 for movement in self.controllers[tls_id].definition.movements
                 for lane in (movement.upstream_lane, movement.downstream_lane)})))
            if state.red_hold_phase is not None or len(state.green_phases) != 2:
                raise ValueError(
                    f"TLS {tls_id} topology is unsupported by binary max-pressure requests: "
                    f"green_phases={state.green_phases}, red_hold={state.red_hold_phase}"
                )
            if tuple(sorted(state.green_phases)) != eligible:
                raise ValueError(
                    f"TLS {tls_id} pressure phases {eligible} do not match controller greens "
                    f"{state.green_phases}"
                )
            target = selected[tls_id]
            if target not in state.green_phases:
                raise ValueError(f"selected max-pressure phase {target} is not a green phase")
            actions[tls_id] = int(target != state.current_green_phase)
        self.last_selected_targets = dict(selected)
        return actions

    def diagnostics(self) -> dict:
        return {"controller_contract": CONTROLLER_CONTRACT_VERSION,
                "definition_hashes": {tls_id: controller.definition.definition_sha256
                                      for tls_id, controller in self.controllers.items()},
                "network_hashes": {tls_id: controller.definition.network_sha256
                                   for tls_id, controller in self.controllers.items()},
                "pressure_equation": "sum_m saturation_m * (q_upstream_m - q_downstream_m)",
                "queue_unit": "last-step halted vehicles"}


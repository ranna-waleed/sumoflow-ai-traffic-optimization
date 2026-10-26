"""Bounded, explicitly labelled computational-cost measurements."""
from __future__ import annotations

import platform
import time
import tracemalloc
import uuid
from typing import Callable

import numpy as np


def measure_callable(function: Callable[[], object], *, repetitions: int,
                     wall_clock_timeout_s: float, label: str) -> dict:
    if repetitions <= 0 or wall_clock_timeout_s <= 0:
        raise ValueError("repetitions and wall-clock timeout must be positive")
    samples = []
    started = time.perf_counter()
    tracemalloc.start()
    try:
        for _ in range(repetitions):
            if time.perf_counter() - started >= wall_clock_timeout_s:
                raise TimeoutError(f"compute pilot exceeded {wall_clock_timeout_s:g}s")
            before = time.perf_counter_ns()
            function()
            samples.append(time.perf_counter_ns() - before)
            if time.perf_counter() - started > wall_clock_timeout_s:
                raise TimeoutError(f"compute pilot exceeded {wall_clock_timeout_s:g}s")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {"schema": "sumoflow-compute-pilot-v1", "label": label,
            "scope": "bounded engineering measurement; no full-scale extrapolation",
            "repetitions": repetitions, "latency_ns": samples,
            "mean_latency_ns": float(np.mean(samples)), "peak_python_bytes": int(peak),
            "wall_clock_s": time.perf_counter() - started,
            "memory_method": "Python allocations sampled by tracemalloc",
            "hardware": {"machine": platform.machine(), "processor": platform.processor()},
            "software": {"python": platform.python_version(), "platform": platform.platform()}}


def measure_traci_fixture(command: list[str], *, maximum_steps: int,
                          wall_clock_timeout_s: float) -> dict:
    """Measure a bounded fixture; step latency includes SUMO plus TraCI round trips."""
    if maximum_steps <= 0 or wall_clock_timeout_s <= 0:
        raise ValueError("maximum_steps and timeout must be positive")
    import traci
    label = f"pilot-{uuid.uuid4().hex}"
    total_start = time.perf_counter()
    tracemalloc.start()
    connection = None
    latencies = []
    try:
        init_start = time.perf_counter()
        traci.start(command, label=label)
        connection = traci.getConnection(label)
        initialization_s = time.perf_counter() - init_start
        simulation_start = time.perf_counter()
        for _ in range(maximum_steps):
            if time.perf_counter() - total_start > wall_clock_timeout_s:
                raise TimeoutError("TraCI compute pilot exceeded its wall-clock limit")
            before = time.perf_counter_ns()
            connection.simulationStep()
            latencies.append(time.perf_counter_ns() - before)
            if connection.simulation.getMinExpectedNumber() == 0:
                break
        simulation_s = time.perf_counter() - simulation_start
        _, peak = tracemalloc.get_traced_memory()
    finally:
        if connection is not None:
            connection.close(True)
        tracemalloc.stop()
    return {"schema": "sumoflow-compute-pilot-v1", "label": "tiny_real_sumo",
            "scope": "bounded fixture only; unsupported for full-scale extrapolation",
            "sumo_initialization_wall_clock_s": initialization_s,
            "simulation_wall_clock_s": simulation_s, "simulation_steps": len(latencies),
            "traci_step_roundtrip_latency_ns": latencies,
            "traci_method": "simulationStep wall time includes SUMO execution and TraCI IPC",
            "peak_python_bytes": int(peak), "memory_method": "tracemalloc Python allocations",
            "hardware": {"machine": platform.machine(), "processor": platform.processor()},
            "software": {"python": platform.python_version(), "platform": platform.platform()}}

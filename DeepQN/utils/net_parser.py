"""
dqn/utils/net_parser.py
-----------------------
Parses tahrirupdated.net.xml to extract:
  - All <tlLogic> phase definitions
  - Each TLS junction's incoming lanes
  - Roundabout edge list

Used at startup to validate the hardcoded TLS config in dqn_config.yaml and
to support future net changes without manual config updates.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


#  Data classes 

@dataclass
class PhaseInfo:
    index: int
    state: str          # e.g. "GGrr", "yyrr", "rrGG"
    duration: float     # default duration in seconds
    is_green: bool = False
    is_yellow: bool = False
    is_red: bool = False

    def __post_init__(self):
        s = self.state.upper()
        self.is_yellow = 'Y' in s and 'G' not in s
        self.is_red    = all(c in ('R',) for c in s)
        self.is_green  = 'G' in s and not self.is_yellow


@dataclass
class TLSInfo:
    tls_id: str
    phases: List[PhaseInfo] = field(default_factory=list)
    incoming_lanes: List[str] = field(default_factory=list)

    # Derived helpers
    @property
    def green_phase_indices(self) -> List[int]:
        return [p.index for p in self.phases if p.is_green]

    @property
    def yellow_phase_indices(self) -> List[int]:
        return [p.index for p in self.phases if p.is_yellow]

    @property
    def red_phase_indices(self) -> List[int]:
        return [p.index for p in self.phases if p.is_red]

    @property
    def is_always_green(self) -> bool:
        """True when the TLS has only one phase that is green (never switches)."""
        return len(self.phases) == 1 and self.phases[0].is_green

    def green_to_yellow_map(self) -> Dict[int, int]:
        """Maps each green phase index to its following yellow phase index."""
        result: Dict[int, int] = {}
        for i, p in enumerate(self.phases):
            if p.is_green:
                # Look for the next yellow phase (may wrap around)
                next_idx = (i + 1) % len(self.phases)
                if self.phases[next_idx].is_yellow:
                    result[p.index] = self.phases[next_idx].index
        return result

    def yellow_to_next_green_map(self) -> Dict[int, int]:
        """Maps each yellow phase index to the green phase that follows it."""
        result: Dict[int, int] = {}
        for i, p in enumerate(self.phases):
            if p.is_yellow:
                next_idx = (i + 1) % len(self.phases)
                # skip any additional red/yellow
                for _ in range(len(self.phases)):
                    candidate = self.phases[next_idx]
                    if candidate.is_green:
                        result[p.index] = candidate.index
                        break
                    next_idx = (next_idx + 1) % len(self.phases)
        return result


#  Parser 

class NetParser:
    """
    Lightweight XML parser for SUMO .net.xml files.

    Usage
    -----
    >>> parser = NetParser("simulation/maps/tahrirupdated.net.xml")
    >>> tls_map = parser.get_all_tls()
    >>> info = tls_map["315744796"]
    >>> print(info.green_phase_indices)   # [0, 2]
    """

    # TLS junctions that the DQN ignores (always-green or not worth controlling)
    ALWAYS_GREEN_IDS = {"6288771435", "96621068"}

    def __init__(self, net_path: str | Path):
        self.net_path = Path(net_path)
        if not self.net_path.exists():
            raise FileNotFoundError(f"net.xml not found: {self.net_path}")
        self._tree = ET.parse(self.net_path)
        self._root = self._tree.getroot()
        logger.info("Parsed net file: %s", self.net_path)

    #  Public API 

    def get_all_tls(self) -> Dict[str, TLSInfo]:
        """Return a dict of tls_id → TLSInfo for all TLS junctions."""
        tls_map: Dict[str, TLSInfo] = {}

        # 1. Extract phase definitions
        for logic_elem in self._root.findall("tlLogic"):
            tls_id = logic_elem.get("id")
            if tls_id is None:
                continue
            info = TLSInfo(tls_id=tls_id)
            for idx, phase_elem in enumerate(logic_elem.findall("phase")):
                state    = phase_elem.get("state", "")
                duration = float(phase_elem.get("duration", 30))
                info.phases.append(PhaseInfo(index=idx, state=state, duration=duration))
            tls_map[tls_id] = info

        # 2. Extract incoming lanes per junction
        for junction_elem in self._root.findall("junction"):
            junc_id = junction_elem.get("id")
            if junc_id not in tls_map:
                continue
            inc_lanes_str = junction_elem.get("incLanes", "")
            if inc_lanes_str:
                tls_map[junc_id].incoming_lanes = inc_lanes_str.split()

        logger.info("Found %d TLS junctions in net.xml", len(tls_map))
        return tls_map

    def get_controllable_tls(self) -> Dict[str, TLSInfo]:
        """Return only TLS junctions that the DQN should control."""
        all_tls = self.get_all_tls()
        controllable = {
            tid: info for tid, info in all_tls.items()
            if tid not in self.ALWAYS_GREEN_IDS and not info.is_always_green
        }
        logger.info(
            "Controllable TLS: %s", list(controllable.keys())
        )
        return controllable

    def get_roundabout_edges(self) -> List[str]:
        """Return the list of edges forming the Tahrir roundabout ring."""
        for rb_elem in self._root.findall("roundabout"):
            edges_str = rb_elem.get("edges", "")
            if edges_str:
                return edges_str.split()
        return []

    def get_roundabout_nodes(self) -> List[str]:
        """Return the junction IDs that form the roundabout."""
        for rb_elem in self._root.findall("roundabout"):
            nodes_str = rb_elem.get("nodes", "")
            if nodes_str:
                return nodes_str.split()
        return []

    def print_summary(self):
        """Pretty-print a summary of all TLS junctions."""
        tls_map = self.get_all_tls()
        print(f"\n{'─'*60}")
        print(f"  Net: {self.net_path.name}")
        print(f"  TLS junctions: {len(tls_map)}")
        print(f"{'─'*60}")
        for tid, info in tls_map.items():
            tag = "[always-green]" if info.is_always_green else ""
            print(f"  {tid} {tag}")
            print(f"    phases       : {len(info.phases)}")
            print(f"    green phases : {info.green_phase_indices}")
            print(f"    incoming lns : {len(info.incoming_lanes)}")
            g2y = info.green_to_yellow_map()
            y2g = info.yellow_to_next_green_map()
            print(f"    green→yellow : {g2y}")
            print(f"    yellow→green : {y2g}")
        print(f"{'─'*60}\n")


#  Config validator 

def validate_tls_config(
    yaml_config: dict,
    net_path: str | Path,
) -> bool:
    """
    Cross-check dqn_config.yaml TLS entries against actual net.xml.
    Logs warnings for any mismatches.  Returns True if all checks pass.
    """
    parser = NetParser(net_path)
    net_tls = parser.get_all_tls()
    ok = True

    for tid, cfg in yaml_config.get("tls_junctions", {}).items():
        if tid not in net_tls:
            logger.error("TLS '%s' in config not found in net.xml", tid)
            ok = False
            continue

        net_info = net_tls[tid]
        cfg_green = set(cfg.get("green_phases", []))
        net_green = set(net_info.green_phase_indices)

        if cfg_green != net_green:
            logger.warning(
                "TLS '%s': config green_phases %s != net.xml green_phases %s",
                tid, cfg_green, net_green,
            )

        cfg_lanes = set(cfg.get("incoming_lanes", []))
        net_lanes = set(net_info.incoming_lanes)
        # SUMO internal lanes (":...") are filtered out
        net_lanes_external = {l for l in net_lanes if not l.startswith(":")}

        extra = cfg_lanes - net_lanes_external
        if extra:
            logger.warning(
                "TLS '%s': config lists lane(s) not in net.xml: %s", tid, extra
            )

    return ok


#  Standalone test 

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    path = sys.argv[1] if len(sys.argv) > 1 else "simulation/maps/tahrirupdated.net.xml"
    NetParser(path).print_summary()
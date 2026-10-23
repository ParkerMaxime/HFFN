"""Runtime counters: what was executed, compiled, allocated and loaded."""
from dataclasses import dataclass, asdict


@dataclass
class Stats:
    ffn_calls: int = 0
    up_dispatches: int = 0
    down_dispatches: int = 0
    queue_waits: int = 0
    queue_peak: int = 0
    library_compiles: int = 0
    workspace_allocations: int = 0
    metadata_writes: int = 0
    metadata_evictions: int = 0
    bank_loads: int = 0
    bank_major_calls: int = 0
    route_major_calls: int = 0

    def snapshot(self):
        return asdict(self)

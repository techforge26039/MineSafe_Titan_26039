from __future__ import annotations
import heapq
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Tuple
from .config import MINE_GRAPH

@dataclass(frozen=True)
class RouteResult:
    path: List[str]
    distance_m: float
    hazard_penalty: float
    total_cost: float
    blocked: bool = False
    reason: str = ""

    def to_dict(self):
        return {
            "path": self.path, "distance_m": round(self.distance_m, 2),
            "hazard_penalty": round(self.hazard_penalty, 2),
            "total_cost": round(self.total_cost, 2),
            "blocked": self.blocked, "reason": self.reason,
        }

def safest_route(start: str, goal: str, blocked_edges: Iterable[Tuple[str, str]] = (),
                 hazard_nodes: Iterable[str] = (), node_penalties: Mapping[str, float] | None = None) -> Tuple[List[str], float]:
    result = route_with_breakdown(start, goal, blocked_edges, hazard_nodes, node_penalties)
    return result.path, result.total_cost

def route_with_breakdown(start: str, goal: str, blocked_edges: Iterable[Tuple[str, str]] = (),
                         hazard_nodes: Iterable[str] = (), node_penalties: Mapping[str, float] | None = None) -> RouteResult:
    if start not in MINE_GRAPH or goal not in MINE_GRAPH:
        return RouteResult([], float("inf"), float("inf"), float("inf"), True, "Unknown node")
    blocked = {frozenset(x) for x in blocked_edges}
    penalties = dict(node_penalties or {})
    for node in hazard_nodes:
        penalties.setdefault(node, 100.0)
    pq = [(0.0, 0.0, 0.0, start, [start])]
    best: Dict[str, float] = {start: 0.0}
    while pq:
        total, distance, hazard, node, path = heapq.heappop(pq)
        if node == goal:
            return RouteResult(path, distance, hazard, total, False, "Safe reachable route")
        if total > best.get(node, float("inf")):
            continue
        for nxt, dist in MINE_GRAPH.get(node, {}).items():
            if frozenset((node, nxt)) in blocked:
                continue
            edge_distance = float(dist)
            node_hazard = max(0.0, float(penalties.get(nxt, 0.0)))
            new_distance = distance + edge_distance
            new_hazard = hazard + node_hazard
            new_total = new_distance + new_hazard
            if new_total < best.get(nxt, float("inf")):
                best[nxt] = new_total
                heapq.heappush(pq, (new_total, new_distance, new_hazard, nxt, path + [nxt]))
    return RouteResult([], float("inf"), float("inf"), float("inf"), True,
                       f"No route from {start} to {goal} under current blockage constraints")

def rank_routes(start: str, goal: str, blocked_edges: Iterable[Tuple[str, str]] = (),
                node_penalties: Mapping[str, float] | None = None) -> List[RouteResult]:
    """Enumerate simple candidate paths and rank them by distance + hazard cost."""
    blocked = {frozenset(x) for x in blocked_edges}
    penalties = dict(node_penalties or {})
    candidates: List[RouteResult] = []
    def dfs(node: str, path: List[str], distance: float, hazard: float):
        if node == goal:
            candidates.append(RouteResult(path[:], distance, hazard, distance + hazard))
            return
        if len(path) > len(MINE_GRAPH) + 1:
            return
        for nxt, dist in MINE_GRAPH.get(node, {}).items():
            if nxt in path or frozenset((node, nxt)) in blocked:
                continue
            nh = hazard + max(0.0, float(penalties.get(nxt, 0.0)))
            dfs(nxt, path + [nxt], distance + float(dist), nh)
    dfs(start, [start], 0.0, 0.0)
    return sorted(candidates, key=lambda r: (r.total_cost, r.hazard_penalty, r.distance_m))[:5]

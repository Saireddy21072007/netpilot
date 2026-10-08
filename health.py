"""Health: rolling-window statistics, per-domain scoring and letter grades.

Scores are tuned against real measurements taken on Indian campus Wi-Fi
and 4G/5G mobile hotspots:
    48 ms avg, 0% loss            -> GO     (score ~93, grade A)
    6% loss                       -> WARN   (grade C/D)
    217 ms avg, wild jitter       -> DEAD   (grade F)

v2 adds: per-domain scores (link / path / dns / radio / bloat) rolled up
into one weighted overall grade.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from prober import PingResult

GO = "GO"
WARN = "WARN"
DEAD = "DEAD"

# Weights for the overall score. Untested/unavailable domains are dropped
# and the rest renormalized, so a light cycle (no bloat test) still scores.
DOMAIN_WEIGHTS = {
    "path": 0.45,     # internet latency/loss/jitter
    "link": 0.25,     # laptop <-> gateway
    "dns": 0.15,      # resolution health
    "radio": 0.15,    # WiFi airwaves environment
}


@dataclass
class HealthSnapshot:
    """Aggregated health of one side (local link or internet path)."""

    label: str
    loss_pct: float = 0.0
    avg_ms: float | None = None
    jitter_ms: float | None = None
    p95_ms: float | None = None
    score: float = 0.0
    verdict: str = DEAD
    samples: int = 0
    extra: dict = field(default_factory=dict)


def _jitter(times: list[float]) -> float | None:
    """Mean absolute difference between consecutive samples (RFC 3550 style)."""
    if len(times) < 2:
        return None
    diffs = [abs(b - a) for a, b in zip(times, times[1:])]
    return sum(diffs) / len(diffs)


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    k = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[k]


def compute_score(
    loss_pct: float,
    avg_ms: float | None,
    jitter_ms: float | None,
    weights: dict,
) -> float:
    s = 100.0
    s -= loss_pct * weights.get("loss_penalty_per_pct", 15)
    if avg_ms is not None:
        baseline = weights.get("rtt_baseline_ms", 60)
        s -= max(0.0, avg_ms - baseline) * weights.get("rtt_penalty_per_ms", 0.3)
    if jitter_ms is not None:
        s -= jitter_ms * weights.get("jitter_penalty_per_ms", 0.5)
    return max(0.0, min(100.0, round(s, 1)))


def letter_grade(score: float, thresholds: dict | None = None) -> str:
    t = thresholds or {"A": 90, "B": 75, "C": 60, "D": 40}
    if score >= t["A"]:
        return "A"
    if score >= t["B"]:
        return "B"
    if score >= t["C"]:
        return "C"
    if score >= t["D"]:
        return "D"
    return "F"


def verdict_of(score: float, thresholds: dict) -> str:
    if score >= thresholds.get("go", 75):
        return GO
    if score >= thresholds.get("warn", 40):
        return WARN
    return DEAD


class RollingHealth:
    """Maintains rolling stats over the last `window_cycles` probe cycles."""

    def __init__(self, label: str, window_cycles: int, weights: dict, thresholds: dict):
        self.label = label
        self.weights = weights
        self.thresholds = thresholds
        self._cycles: deque[list[PingResult]] = deque(maxlen=window_cycles)

    def add_cycle(self, results: list[PingResult]) -> None:
        self._cycles.append(results)

    def snapshot(self) -> HealthSnapshot:
        flat: list[PingResult] = [r for cycle in self._cycles for r in cycle]
        snap = HealthSnapshot(label=self.label)
        if not flat:
            return snap

        total_sent = sum(r.sent for r in flat)
        total_recv = sum(r.received for r in flat)
        snap.samples = total_recv
        snap.loss_pct = 100.0 * (total_sent - total_recv) / total_sent if total_sent else 100.0

        all_times = [t for r in flat for t in r.times_ms]
        snap.avg_ms = sum(all_times) / len(all_times) if all_times else None
        snap.p95_ms = _percentile(all_times, 95) if all_times else None

        # Jitter is measured per host (consecutive pings to the same host),
        # then averaged across hosts for a stable number.
        jitters = [_jitter(r.times_ms) for r in flat if len(r.times_ms) >= 2]
        jitters = [j for j in jitters if j is not None]
        snap.jitter_ms = sum(jitters) / len(jitters) if jitters else None

        snap.score = compute_score(snap.loss_pct, snap.avg_ms, snap.jitter_ms, self.weights)
        snap.verdict = verdict_of(snap.score, self.thresholds)
        return snap


@dataclass
class Overall:
    """Roll-up of every domain into one number + explanation."""

    domains: dict[str, float | None] = field(default_factory=dict)
    overall_score: float = 0.0
    verdict: str = DEAD

    def grade(self, thresholds: dict | None = None) -> str:
        return letter_grade(self.overall_score, thresholds)

    def weakest(self) -> tuple[str, float] | None:
        scored = {k: v for k, v in self.domains.items() if v is not None}
        if not scored:
            return None
        k = min(scored, key=scored.get)
        return k, scored[k]


def roll_up(
    path: HealthSnapshot,
    link: HealthSnapshot | None,
    dns_score: float | None,
    radio_score: float | None,
    bloat_score: float | None,
    thresholds: dict,
) -> Overall:
    domains: dict[str, float | None] = {"path": path.score}

    if link is not None and link.samples > 0:
        domains["link"] = link.score
    else:
        domains["link"] = None  # gateway unmeasurable (ICMP-blocked, VPN, wired)

    domains["dns"] = dns_score
    domains["radio"] = radio_score
    domains["bloat"] = bloat_score  # None on light cycles — ignored below

    weights = dict(DOMAIN_WEIGHTS)
    if bloat_score is not None:
        weights = dict(DOMAIN_WEIGHTS)
        weights["bloat"] = 0.20
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

    available = {k: v for k, v in domains.items() if v is not None and k in weights}
    if available:
        wsum = sum(weights[k] for k in available)
        overall = sum(v * weights[k] for k, v in available.items()) / wsum if wsum else 0.0
    else:
        overall = 0.0

    ov = Overall(domains=domains, overall_score=round(max(0.0, min(100.0, overall)), 1))
    ov.verdict = verdict_of(ov.overall_score, thresholds)
    return ov

   
def diagnose(
    gateway_snap: HealthSnapshot | None,
    net_snap: HealthSnapshot,
    gateway_icmp_blocked: bool = False,
) -> str:
    """Plain-English verdict on WHICH side of the path is the problem."""
    local_bad = gateway_snap is not None and gateway_snap.verdict != GO and not gateway_icmp_blocked
    net_bad = net_snap.verdict != GO

    if net_bad and local_bad:
        return "Laptop<->router link AND internet path both unstable. Move closer to the AP / try another network."
    if net_bad:
        if gateway_icmp_blocked:
            return "Internet path is congested (gateway ignores ping, can't isolate further). Switch network or wait."
        return "Router/WiFi link is fine, but the upstream internet path is congested (ISP/campus). Switch network or wait."
    if local_bad:
        return "Local WiFi link to the router is unstable (distance/interference), but internet beyond it is okay."
    if gateway_icmp_blocked:
        return "All clear. (Gateway AP ignores ping; internet path is healthy.)"
    return "All clear."

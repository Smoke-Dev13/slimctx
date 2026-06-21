"""JSON smart compressor with MinHash-based stratified sampling.

Targets JSON arrays of homogeneous dict records (shared schema, differing values).
Replaces naive head-N truncation with:

  1. Per-field statistics: type detection, numeric distribution, cardinality.
  2. Anomaly flagging: z-score > 2 on numeric fields; categorical values
     appearing in < 5% of records.
  3. MinHash clustering: groups records by discretised feature similarity;
     one representative per cluster.
  4. Query-aware aggressiveness: "count/aggregate" → aggressive;
     "find/unusual/specific" → conservative.
  5. _sample_meta footer: {"total": N, "shown": M, "strategy": ...}.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from typing import Any

import structlog

from contextly.compressors.base import Compressor, CompressResult

logger = structlog.get_logger(__name__)

# ── Tuning constants ───────────────────────────────────────────────────────────

_NUM_PERM: int = 64
_LSH_THRESHOLD: float = 0.3
_NUM_BUCKETS: int = 5
_Z_THRESHOLD: float = 2.0
_RARE_THRESHOLD: float = 0.05
_MIN_RECORDS: int = 10
_MIN_SHARED_KEYS: int = 3
_HIGH_CARD_RATIO: float = 0.7

_AGGRESSIVE_KEYWORDS: frozenset[str] = frozenset(
    ["count", "total", "sum", "average", "aggregate", "how many", "number of", "statistics"]
)
_CONSERVATIVE_KEYWORDS: frozenset[str] = frozenset(
    ["unusual", "outlier", "anomal", "rare", "specific", "find", "identify", "which", "who"]
)


# ── Field statistics ───────────────────────────────────────────────────────────


def _compute_percentile_buckets(sorted_vals: list[float], n_buckets: int) -> list[float]:
    """Return (n_buckets - 1) boundary values at even percentile intervals."""
    n = len(sorted_vals)
    if n == 0:
        return []
    return [sorted_vals[min(int(i * n / n_buckets), n - 1)] for i in range(1, n_buckets)]


def _get_bucket(value: float, thresholds: list[float]) -> int:
    """Return the 0-based bucket index for *value* given sorted *thresholds*."""
    for i, t in enumerate(thresholds):
        if value <= t:
            return i
    return len(thresholds)


def _analyze_fields(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Compute per-field type and statistics over *records*.

    Returns a dict keyed by field name. Each entry has at minimum:
      "type": "numeric" | "categorical" | "boolean" | "high_cardinality" | "other"
    Plus type-specific keys:
      numeric      → mean, stdev, buckets (list[float]), values (list[float])
      categorical  → freqs ({str: float}), cardinality (int)
    """
    n = len(records)
    field_values: dict[str, list[Any]] = {}
    for record in records:
        for key, val in record.items():
            field_values.setdefault(key, []).append(val)

    stats: dict[str, dict[str, Any]] = {}
    for field, vals in field_values.items():
        nums = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
        bools = [v for v in vals if isinstance(v, bool)]
        strs = [v for v in vals if isinstance(v, str)]

        if len(nums) >= len(vals) * 0.5:
            mean = statistics.mean(nums)
            stdev = statistics.stdev(nums) if len(nums) > 1 else 0.0
            sorted_nums = sorted(nums)
            stats[field] = {
                "type": "numeric",
                "mean": mean,
                "stdev": stdev,
                "buckets": _compute_percentile_buckets(sorted_nums, _NUM_BUCKETS),
                "values": nums,
            }
        elif bools:
            stats[field] = {"type": "boolean"}
        elif strs:
            counts: Counter[str] = Counter(strs)
            cardinality = len(counts)
            if cardinality > n * _HIGH_CARD_RATIO:
                stats[field] = {"type": "high_cardinality"}
            else:
                stats[field] = {
                    "type": "categorical",
                    "cardinality": cardinality,
                    "freqs": {v: c / n for v, c in counts.items()},
                }
        else:
            stats[field] = {"type": "other"}

    return stats


# ── Anomaly detection ─────────────────────────────────────────────────────────


def _find_anomalies(
    records: list[dict[str, Any]],
    field_stats: dict[str, dict[str, Any]],
) -> set[int]:
    """Return indices of records that are anomalous in at least one field.

    Anomaly criteria (configurable via module constants):
      numeric     → |z-score| > _Z_THRESHOLD  (default 2.0)
      categorical → value frequency < _RARE_THRESHOLD (default 5%)
    """
    anomalies: set[int] = set()
    for i, record in enumerate(records):
        for field, value in record.items():
            stat = field_stats.get(field, {})
            ftype = stat.get("type")
            if ftype == "numeric":
                stdev = stat["stdev"]
                if stdev > 0 and isinstance(value, (int, float)) and not isinstance(value, bool):
                    z = abs(value - stat["mean"]) / stdev
                    if z > _Z_THRESHOLD:
                        anomalies.add(i)
            elif ftype == "categorical":
                freqs: dict[str, float] = stat.get("freqs", {})
                freq = freqs.get(str(value), 0.0)
                if 0.0 < freq < _RARE_THRESHOLD:
                    anomalies.add(i)
    return anomalies


# ── MinHash clustering ─────────────────────────────────────────────────────────


def _minhash_record(record: dict[str, Any], field_stats: dict[str, dict[str, Any]]) -> Any:
    """Build a MinHash fingerprint for *record* from discretised features.

    High-cardinality fields (IDs, unique strings) are skipped so that records
    with the same *pattern* cluster together regardless of unique identifiers.
    Numeric values are discretised into percentile buckets.
    """
    from datasketch import MinHash

    m: Any = MinHash(num_perm=_NUM_PERM)
    for key in sorted(record.keys()):
        stat = field_stats.get(key, {})
        ftype = stat.get("type", "other")
        if ftype in ("high_cardinality", "other"):
            continue
        value = record.get(key)
        if ftype == "numeric" and isinstance(value, (int, float)) and not isinstance(value, bool):
            bucket = _get_bucket(float(value), stat.get("buckets", []))
            feature = f"{key}:b{bucket}"
        elif ftype == "categorical":
            feature = f"{key}:{value}"
        elif ftype == "boolean":
            feature = f"{key}:{value}"
        else:
            continue
        m.update(feature.encode("utf-8"))
    return m


def _cluster_with_minhash(minhashes: list[Any]) -> list[list[int]]:
    """Cluster record indices using MinHash LSH + union-find.

    Records whose Jaccard similarity of feature sets exceeds _LSH_THRESHOLD
    are merged into the same cluster. Returns a list of clusters, each being
    a list of record indices.
    """
    from datasketch import MinHashLSH

    n = len(minhashes)
    if n == 0:
        return []

    lsh: Any = MinHashLSH(threshold=_LSH_THRESHOLD, num_perm=_NUM_PERM)
    for i, m in enumerate(minhashes):
        lsh.insert(str(i), m)

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, m in enumerate(minhashes):
        similar: list[str] = lsh.query(m)
        for key in similar:
            j = int(key)
            if j != i:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    return list(groups.values())


# ── Query-aware aggressiveness ─────────────────────────────────────────────────


def _aggressiveness_from_query(query: str) -> float:
    """Return an aggressiveness factor in [0, 1].  Higher → fewer records kept.

    Heuristics based on common LLM query patterns:
      "count / aggregate"  → 0.9  (summary; full detail not needed)
      "find / unusual"     → 0.3  (detail matters; preserve outliers)
      (default)            → 0.7
    """
    if not query:
        return 0.7
    q = query.lower()
    if any(kw in q for kw in _CONSERVATIVE_KEYWORDS):
        return 0.3
    if any(kw in q for kw in _AGGRESSIVE_KEYWORDS):
        return 0.9
    return 0.7


def _target_sample_size(total: int, aggressiveness: float) -> int:
    """Compute the target number of records to retain.

    Bounds: at least 2% of total (min 3), at most 15% (min 10).
    Aggressiveness linearly interpolates between max_keep and min_keep.
    """
    min_keep = max(3, math.ceil(total * 0.02))
    max_keep = max(10, math.ceil(total * 0.15))
    target = round(max_keep - (max_keep - min_keep) * aggressiveness)
    return max(min_keep, min(max_keep, target))


# ── Helpers ───────────────────────────────────────────────────────────────────


def _is_homogeneous(records: list[dict[str, Any]]) -> bool:
    """Return True if all records share at least _MIN_SHARED_KEYS common keys."""
    if not records:
        return False
    common = set(records[0].keys())
    for r in records[1:]:
        common &= r.keys()
        if len(common) < _MIN_SHARED_KEYS:
            return False
    return len(common) >= _MIN_SHARED_KEYS


def _make_passthrough(content: str, name: str) -> CompressResult:
    length = len(content)
    return CompressResult(
        content=content,
        original_length=length,
        compressed_length=length,
        compressor_name=name,
    )


# ── Compressor ────────────────────────────────────────────────────────────────


class JsonSmartCompressor(Compressor):
    """Stratified sampling with anomaly preservation for homogeneous JSON arrays.

    Detection (should_apply): O(1) heuristic — checks opening bracket + brace.

    Compression pipeline:
      1. Parse JSON; bail to passthrough if not a list of dicts.
      2. Check homogeneity (>= _MIN_SHARED_KEYS shared keys, >= _MIN_RECORDS).
      3. Analyse per-field statistics (type, numeric distribution, cardinality).
      4. Flag anomalies (z-score > 2, rare categoricals < 5%).
      5. Build MinHash per record from discretised non-ID features.
      6. Cluster via LSH + union-find; pick one representative per cluster.
      7. Keep all anomalies; add cluster reps until target count is reached.
      8. Append {"_sample_meta": {...}} and serialise with compact separators.
    """

    @property
    def name(self) -> str:
        return "json_smart"

    def should_apply(self, content: str, query: str = "") -> bool:
        """Return True if content looks like a JSON array of objects."""
        stripped = content.lstrip()
        return stripped.startswith("[") and "{" in stripped[:500]

    def compress(self, content: str, query: str = "") -> CompressResult:
        """Apply stratified sampling with anomaly preservation.

        Falls back to passthrough if the content is not a compressible
        homogeneous JSON array.
        """
        original_length = len(content)

        try:
            data: Any = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            return _make_passthrough(content, self.name)

        if not isinstance(data, list):
            return _make_passthrough(content, self.name)

        records: list[dict[str, Any]] = [r for r in data if isinstance(r, dict)]
        if len(records) < _MIN_RECORDS or not _is_homogeneous(records):
            return _make_passthrough(content, self.name)

        log = logger.bind(total=len(records), query_preview=query[:80])
        log.debug("json_smart_compressing")

        field_stats = _analyze_fields(records)

        # Bail if every field is a unique ID (nothing useful to cluster on).
        skip = {"high_cardinality", "other"}
        informative = [f for f, s in field_stats.items() if s.get("type") not in skip]
        if not informative:
            return _make_passthrough(content, self.name)

        anomaly_indices = _find_anomalies(records, field_stats)

        try:
            minhashes = [_minhash_record(r, field_stats) for r in records]
            clusters = _cluster_with_minhash(minhashes)
        except Exception:
            log.warning("json_smart_clustering_failed", exc_info=True)
            return _make_passthrough(content, self.name)

        aggressiveness = _aggressiveness_from_query(query)
        target = _target_sample_size(len(records), aggressiveness)

        selected: set[int] = set(anomaly_indices)
        for cluster in sorted(clusters, key=lambda c: -len(c)):
            if len(selected) >= target:
                break
            rep = cluster[0]
            if rep not in selected:
                selected.add(rep)

        selected_records = [records[i] for i in sorted(selected)]
        meta_record = {
            "_sample_meta": {
                "total": len(records),
                "shown": len(selected_records),
                "strategy": "stratified+outliers",
            }
        }
        output: list[Any] = [*selected_records, meta_record]
        compressed_content = json.dumps(output, separators=(",", ":"))
        compressed_length = len(compressed_content)

        log.info(
            "json_smart_compressed",
            total=len(records),
            selected=len(selected_records),
            anomalies=len(anomaly_indices),
            clusters=len(clusters),
            ratio=round(compressed_length / original_length, 3),
        )

        return CompressResult(
            content=compressed_content,
            original_length=original_length,
            compressed_length=compressed_length,
            compressor_name=self.name,
            metadata={
                "total_records": len(records),
                "selected_records": len(selected_records),
                "anomaly_count": len(anomaly_indices),
                "cluster_count": len(clusters),
                "aggressiveness": round(aggressiveness, 2),
            },
        )

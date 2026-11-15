#!/usr/bin/env python3
"""
Day 38: Benchmark JSONL Aggregator with Documented Nearest-Rank Percentiles,
Outlier Preservation, Separate Failure Accounting, and Run Manifest Metadata.
"""
import argparse
import json
import math
import os
import platform
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


PERCENTILE_CONVENTION = {
    "method": "nearest-rank",
    "formula": "sorted_values[min(n - 1, max(0, ceil((q / 100.0) * n) - 1))]",
    "outlier_policy": "strict_preservation_no_trimming_or_winsorization",
    "failure_handling": "excluded_from_latency_percentiles_reported_in_failure_rate",
}


def nearest_rank_percentile(values: List[float], q: float) -> Optional[float]:
    """Compute exact nearest-rank percentile without trimming outliers."""
    if not values:
        return None
    if not (0 < q <= 100):
        raise ValueError(f"Percentile q must be in (0, 100], got {q}")
    ordered = sorted(values)
    n = len(ordered)
    rank_idx = min(n - 1, max(0, math.ceil((q / 100.0) * n) - 1))
    return round(float(ordered[rank_idx]), 3)


def summarize_distribution(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
            "mean": None,
        }
    return {
        "count": len(values),
        "min": round(float(min(values)), 3),
        "p50": nearest_rank_percentile(values, 50),
        "p95": nearest_rank_percentile(values, 95),
        "p99": nearest_rank_percentile(values, 99),
        "max": round(float(max(values)), 3),
        "mean": round(float(statistics.mean(values)), 3),
    }


def summarize_records(records: List[Dict[str, Any]], source_file: str = "in-memory") -> Dict[str, Any]:
    warmup_records = [r for r in records if r.get("phase") == "warmup"]
    measured_records = [r for r in records if r.get("phase") == "measured"]

    if not measured_records:
        raise ValueError("No records with phase='measured' found; cannot compute benchmark summary.")

    ok_records = [r for r in measured_records if r.get("status") == "ok"]
    err_records = [r for r in measured_records if r.get("status") != "ok"]

    failure_rate = round(len(err_records) / len(measured_records), 4)
    error_breakdown = dict(Counter(r.get("error_type") or "unknown_error" for r in err_records))

    ttft_vals = [float(r["ttft_ms"]) for r in ok_records if r.get("ttft_ms") is not None]
    itl_key = "mean_itl_ms" if any("mean_itl_ms" in r for r in ok_records) else "tpot_ms"
    itl_vals = [float(r[itl_key]) for r in ok_records if r.get(itl_key) is not None]
    lat_vals = [float(r["total_latency_ms"]) for r in ok_records if r.get("total_latency_ms") is not None]

    tps_key = "output_tokens_per_sec" if any("output_tokens_per_sec" in r for r in ok_records) else "tokens_per_sec"
    tps_vals = [float(r[tps_key]) for r in ok_records if r.get(tps_key) is not None]

    total_output_tokens = sum(int(r.get("output_tokens") or r.get("completion_tokens") or 0) for r in ok_records)
    total_wall_sec = sum(float(r.get("total_latency_ms") or 0.0) for r in ok_records) / 1000.0
    aggregate_tps = round(total_output_tokens / total_wall_sec, 3) if total_wall_sec > 0 else None

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": source_file,
        "environment": {
            "os": platform.platform(),
            "python_version": sys.version.split()[0],
            "cpu_cores": os.cpu_count(),
        },
        "percentile_convention": PERCENTILE_CONVENTION,
        "sample_counts": {
            "total_jsonl_rows": len(records),
            "warmup_excluded": len(warmup_records),
            "measured_total": len(measured_records),
            "measured_ok": len(ok_records),
            "measured_error": len(err_records),
        },
        "reliability": {
            "failure_rate": failure_rate,
            "success_rate": round(1.0 - failure_rate, 4),
            "error_breakdown": error_breakdown,
        },
        "throughput": {
            "total_output_tokens": total_output_tokens,
            "aggregate_output_tokens_per_sec": aggregate_tps,
            "per_request_tps_distribution": summarize_distribution(tps_vals),
        },
        "latency_ms": {
            "ttft_ms": summarize_distribution(ttft_vals),
            "itl_ms": summarize_distribution(itl_vals),
            "total_latency_ms": summarize_distribution(lat_vals),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Aggregate benchmark JSONL into percentiles and failure rate summary.")
    parser.add_argument("--input", default="artifacts/day37/day37-latency.jsonl", help="Path to input JSONL file")
    parser.add_argument("--output", default="artifacts/day38/day38-summary.json", help="Path to output summary JSON")
    args = parser.parse_args()

    records = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    summary = summarize_records(records, source_file=args.input)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary written to {args.output}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

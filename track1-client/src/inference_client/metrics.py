"""
Day 37: Exact Stream Latency & Throughput Instrumentation (TTFT, ITL, E2E TPS).
"""
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
import statistics


@dataclass
class StreamMetricsRecord:
    request_id: str
    phase: str  # "warmup" | "measured"
    status: str  # "ok" | "error"
    prompt_tokens: Optional[int]
    output_tokens: Optional[int]
    ttft_ms: Optional[float]
    token_deltas_ms: List[float]
    mean_itl_ms: Optional[float]
    total_latency_ms: float
    output_tokens_per_sec: Optional[float]
    decode_tokens_per_sec: Optional[float]
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_stream_metrics(
    request_id: str,
    phase: str,
    t_start_sec: float,
    token_timestamps_sec: List[float],
    t_end_sec: float,
    prompt_tokens: int,
    output_tokens: Optional[int] = None,
) -> StreamMetricsRecord:
    """
    Compute exact TTFT, inter-token deltas, mean ITL, total latency, and throughput
    from monotonic timestamps (in seconds).
    """
    if not token_timestamps_sec:
        raise ValueError("token_timestamps_sec must contain at least one token arrival timestamp")
    if t_start_sec > token_timestamps_sec[0]:
        raise ValueError("First token arrival cannot precede request start time")
    if token_timestamps_sec[-1] > t_end_sec:
        raise ValueError("Stream end time cannot precede final token arrival")

    n_tokens = output_tokens if output_tokens is not None else len(token_timestamps_sec)
    ttft_ms = (token_timestamps_sec[0] - t_start_sec) * 1000.0
    total_latency_ms = (t_end_sec - t_start_sec) * 1000.0

    deltas_ms = [
        round((token_timestamps_sec[i] - token_timestamps_sec[i - 1]) * 1000.0, 6)
        for i in range(1, len(token_timestamps_sec))
    ]

    mean_itl_ms = statistics.mean(deltas_ms) if deltas_ms else 0.0

    e2e_tps = (n_tokens / (total_latency_ms / 1000.0)) if total_latency_ms > 0 else 0.0
    decode_duration_sec = (token_timestamps_sec[-1] - token_timestamps_sec[0])
    decode_tps = ((n_tokens - 1) / decode_duration_sec) if (n_tokens > 1 and decode_duration_sec > 0) else 0.0

    return StreamMetricsRecord(
        request_id=request_id,
        phase=phase,
        status="ok",
        prompt_tokens=prompt_tokens,
        output_tokens=n_tokens,
        ttft_ms=round(ttft_ms, 3),
        token_deltas_ms=[round(d, 3) for d in deltas_ms],
        mean_itl_ms=round(mean_itl_ms, 3),
        total_latency_ms=round(total_latency_ms, 3),
        output_tokens_per_sec=round(e2e_tps, 3),
        decode_tokens_per_sec=round(decode_tps, 3),
    )


def summarize_measured_records(records: List[StreamMetricsRecord]) -> Dict[str, Any]:
    """Summarize only measured + successful records while reporting failure rate separately."""
    measured = [r for r in records if r.phase == "measured"]
    ok_records = [r for r in measured if r.status == "ok"]
    err_records = [r for r in measured if r.status != "ok"]

    if not ok_records:
        return {
            "measured_total": len(measured),
            "measured_ok": 0,
            "measured_errors": len(err_records),
            "failure_rate": round(len(err_records) / max(1, len(measured)), 4),
            "mean_ttft_ms": None,
            "mean_itl_ms": None,
            "mean_total_latency_ms": None,
            "mean_output_tokens_per_sec": None,
        }

    return {
        "measured_total": len(measured),
        "measured_ok": len(ok_records),
        "measured_errors": len(err_records),
        "failure_rate": round(len(err_records) / len(measured), 4),
        "mean_ttft_ms": round(statistics.mean(r.ttft_ms for r in ok_records), 3),
        "mean_itl_ms": round(statistics.mean(r.mean_itl_ms for r in ok_records), 3),
        "mean_total_latency_ms": round(statistics.mean(r.total_latency_ms for r in ok_records), 3),
        "mean_output_tokens_per_sec": round(statistics.mean(r.output_tokens_per_sec for r in ok_records), 3),
    }

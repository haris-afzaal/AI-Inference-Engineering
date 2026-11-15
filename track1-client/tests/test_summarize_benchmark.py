import unittest
from tools.summarize_benchmark import nearest_rank_percentile, summarize_records


class TestSummarizeBenchmark(unittest.TestCase):
    def test_synthetic_outlier_preserved_in_tail(self):
        # 19 fast requests (~100 ms) + 1 severe tail outlier (2500 ms)
        records = [
            {"phase": "warmup", "status": "ok", "ttft_ms": 900.0, "mean_itl_ms": 50.0, "total_latency_ms": 1800.0, "output_tokens": 10, "output_tokens_per_sec": 5.5}
        ]
        for i in range(19):
            records.append({
                "phase": "measured", "status": "ok",
                "ttft_ms": 30.0 + i, "mean_itl_ms": 10.0, "total_latency_ms": 100.0 + i,
                "output_tokens": 10, "output_tokens_per_sec": 90.0
            })
        # Inject synthetic 2500ms tail spike
        records.append({
            "phase": "measured", "status": "ok",
            "ttft_ms": 1200.0, "mean_itl_ms": 80.0, "total_latency_ms": 2500.0,
            "output_tokens": 10, "output_tokens_per_sec": 4.0
        })

        summary = summarize_records(records, source_file="synthetic_outlier_test")
        self.assertEqual(summary["sample_counts"]["warmup_excluded"], 1)
        self.assertEqual(summary["sample_counts"]["measured_total"], 20)
        # p50 stays near normal (~109ms), while p95/p99 strictly capture the 2500ms outlier
        self.assertEqual(summary["latency_ms"]["total_latency_ms"]["p50"], 109.0)
        self.assertEqual(summary["latency_ms"]["total_latency_ms"]["p95"], 118.0)
        self.assertEqual(summary["latency_ms"]["total_latency_ms"]["p99"], 2500.0)
        self.assertEqual(summary["latency_ms"]["ttft_ms"]["p99"], 1200.0)

    def test_fast_failing_errors_excluded_from_latency(self):
        records = [
            {"phase": "measured", "status": "ok", "ttft_ms": 40.0, "mean_itl_ms": 12.0, "total_latency_ms": 120.0, "output_tokens": 8, "output_tokens_per_sec": 66.6},
            {"phase": "measured", "status": "error", "error_type": "HTTPError_503", "ttft_ms": None, "mean_itl_ms": None, "total_latency_ms": 3.0, "output_tokens": None, "output_tokens_per_sec": None},
        ]
        summary = summarize_records(records)
        self.assertEqual(summary["reliability"]["failure_rate"], 0.5)
        self.assertEqual(summary["reliability"]["error_breakdown"], {"HTTPError_503": 1})
        # 3.0ms fast failure must NOT pull down p50 latency
        self.assertEqual(summary["latency_ms"]["total_latency_ms"]["p50"], 120.0)


if __name__ == "__main__":
    unittest.main()

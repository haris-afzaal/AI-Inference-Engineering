import unittest
from inference_client.metrics import (
    StreamMetricsRecord,
    compute_stream_metrics,
    summarize_measured_records,
)


class TestStreamMetrics(unittest.TestCase):
    def test_hand_computed_three_token_trace(self):
        # Hand-computed trace:
        # t_start=0.000s, tok_1=0.100s (TTFT=100ms), tok_2=0.130s (30ms), tok_3=0.170s (40ms)
        # mean_itl = (30 + 40)/2 = 35.0ms, total = 170.0ms, E2E TPS = 3/0.170 = 17.647
        rec = compute_stream_metrics(
            request_id="req-3tok",
            phase="measured",
            t_start_sec=0.000,
            token_timestamps_sec=[0.100, 0.130, 0.170],
            t_end_sec=0.170,
            prompt_tokens=12,
            output_tokens=3,
        )
        self.assertEqual(rec.ttft_ms, 100.0)
        self.assertEqual(rec.token_deltas_ms, [30.0, 40.0])
        self.assertEqual(rec.mean_itl_ms, 35.0)
        self.assertEqual(rec.total_latency_ms, 170.0)
        self.assertAlmostEqual(rec.output_tokens_per_sec, 17.647, places=3)
        self.assertAlmostEqual(rec.decode_tokens_per_sec, 28.571, places=3)

    def test_single_token_stream_edge_case(self):
        rec = compute_stream_metrics("req-1tok", "measured", 1.000, [1.080], 1.085, 5, 1)
        self.assertEqual(rec.ttft_ms, 80.0)
        self.assertEqual(rec.token_deltas_ms, [])
        self.assertEqual(rec.mean_itl_ms, 0.0)
        self.assertEqual(rec.decode_tokens_per_sec, 0.0)

    def test_warmup_and_error_isolation_in_summary(self):
        warmup = compute_stream_metrics("w-0", "warmup", 0.0, [0.500, 0.600], 0.600, 10, 2)
        ok_run = compute_stream_metrics("m-0", "measured", 0.0, [0.100, 0.130, 0.170], 0.170, 12, 3)
        err_run = StreamMetricsRecord(
            request_id="m-1", phase="measured", status="error",
            prompt_tokens=None, output_tokens=None, ttft_ms=None,
            token_deltas_ms=[], mean_itl_ms=None, total_latency_ms=12.5,
            output_tokens_per_sec=None, decode_tokens_per_sec=None,
            error_type="HTTPError_503", error_message="Engine overloaded",
        )
        summary = summarize_measured_records([warmup, ok_run, err_run])
        self.assertEqual(summary["measured_total"], 2)
        self.assertEqual(summary["measured_ok"], 1)
        self.assertEqual(summary["failure_rate"], 0.5)
        self.assertEqual(summary["mean_ttft_ms"], 100.0)
        self.assertEqual(summary["mean_itl_ms"], 35.0)


if __name__ == "__main__":
    unittest.main()

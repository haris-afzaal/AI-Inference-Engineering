import unittest
from tools.memory_budget import WorkloadSpec, estimate_memory_budget


class TestMemoryBudget(unittest.TestCase):
    def test_llama3_8b_fp16_vs_int8_on_t4_15gib_budget(self):
        fp16_spec = WorkloadSpec(
            model_name="Llama-3-8B",
            params_billions=8.0,
            weight_dtype="fp16",
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            kv_dtype="fp16",
            prompt_tokens=512,
            output_tokens=512,
            concurrency=1,
            runtime_overhead_gib=0.60,
            device_budget_gib=15.0,
            memory_bandwidth_gbps=300.0,
        )
        res_fp16 = estimate_memory_budget(fp16_spec)
        self.assertEqual(res_fp16["kv_bytes_per_token"], 131072)
        self.assertEqual(res_fp16["seq_len"], 1024)
        self.assertAlmostEqual(res_fp16["kv_cache_gib"], 0.125, places=4)
        # 14.9012 GiB weights + 0.125 GiB KV + 0.60 GiB overhead = 15.6262 GiB > 15.0 GiB T4 usable VRAM
        self.assertFalse(res_fp16["fits_in_budget"])
        self.assertLess(res_fp16["safety_margin_gib"], 0.0)

        int8_spec = WorkloadSpec(
            model_name="Llama-3-8B",
            params_billions=8.0,
            weight_dtype="int8",
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            kv_dtype="fp16",
            prompt_tokens=2048,
            output_tokens=2048,
            concurrency=8,
            runtime_overhead_gib=0.60,
            device_budget_gib=15.0,
            memory_bandwidth_gbps=300.0,
        )
        res_int8 = estimate_memory_budget(int8_spec)
        self.assertTrue(res_int8["fits_in_budget"])
        self.assertGreater(res_int8["safety_margin_gib"], 2.9)

    def test_llama3_8b_fp16_oom_at_4k_context_c4(self):
        spec = WorkloadSpec(
            model_name="Llama-3-8B",
            params_billions=8.0,
            weight_dtype="fp16",
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            kv_dtype="fp16",
            prompt_tokens=2048,
            output_tokens=2048,
            concurrency=4,
            runtime_overhead_gib=0.60,
            device_budget_gib=15.0,
            memory_bandwidth_gbps=300.0,
        )
        res = estimate_memory_budget(spec)
        self.assertAlmostEqual(res["kv_cache_gib"], 2.0, places=4)
        self.assertFalse(res["fits_in_budget"])
        self.assertLess(res["safety_margin_gib"], 0.0)

    def test_invalid_dtype_raises_value_error(self):
        bad_spec = WorkloadSpec(
            model_name="Llama-3-8B",
            params_billions=8.0,
            weight_dtype="fp7",
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            kv_dtype="fp16",
            prompt_tokens=512,
            output_tokens=512,
            concurrency=1,
            runtime_overhead_gib=0.60,
            device_budget_gib=15.0,
            memory_bandwidth_gbps=300.0,
        )
        with self.assertRaises(ValueError):
            estimate_memory_budget(bad_spec)

    def test_zero_head_dim_raises_value_error(self):
        bad_spec = WorkloadSpec(
            model_name="Llama-3-8B",
            params_billions=8.0,
            weight_dtype="fp16",
            num_layers=32,
            num_kv_heads=8,
            head_dim=0,
            kv_dtype="fp16",
            prompt_tokens=512,
            output_tokens=512,
            concurrency=1,
            runtime_overhead_gib=0.60,
            device_budget_gib=15.0,
            memory_bandwidth_gbps=300.0,
        )
        with self.assertRaises(ValueError):
            estimate_memory_budget(bad_spec)


if __name__ == "__main__":
    unittest.main()

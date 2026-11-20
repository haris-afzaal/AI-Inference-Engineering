import csv
import statistics
import time
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DecoderBlock(nn.Module):
    def __init__(self, d_model=1024, nhead=16):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x

def benchmark_config(seq_len, dtype, decode_steps=12, warmups=2, reps=5):
    torch.manual_seed(48)
    model = nn.Sequential(*[DecoderBlock() for _ in range(4)]).to(device=device, dtype=dtype).eval()
    torch.manual_seed(99)
    prompt = torch.randn(1, seq_len, 1024, device=device, dtype=dtype)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for _ in range(warmups):
            h = model(prompt)
            tok = h[:, -1:, :]
            for _ in range(decode_steps):
                tok = model(tok)
            if device.type == "cuda":
                torch.cuda.synchronize()

        latencies_ms = []
        last_logits = None
        for _ in range(reps):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            h = model(prompt)
            tok = h[:, -1:, :]
            for _ in range(decode_steps):
                tok = model(tok)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)
            last_logits = tok.float().cpu()

    peak_mem_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda"
        else sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    )
    p50_ms = statistics.median(latencies_ms)
    mean_ms = statistics.mean(latencies_ms)
    tps = (seq_len + decode_steps) / (p50_ms / 1000.0)
    return {
        "seq_len": seq_len,
        "dtype": str(dtype).replace("torch.", ""),
        "p50_ms": round(p50_ms, 3),
        "mean_ms": round(mean_ms, 3),
        "throughput_tps": round(tps, 2),
        "peak_mem_mb": round(peak_mem_mb, 2),
        "logits": last_logits,
    }

rows = []
for seq_len in [128, 512]:
    base = benchmark_config(seq_len, torch.float32)
    cand = benchmark_config(seq_len, torch.float16)

    max_abs_diff = float((base["logits"] - cand["logits"]).abs().max().item())
    top1_match = bool(base["logits"].argmax().item() == cand["logits"].argmax().item())
    has_nan = bool(torch.isnan(cand["logits"]).any().item() or torch.isinf(cand["logits"]).any().item())
    lat_delta_pct = round(((cand["p50_ms"] - base["p50_ms"]) / base["p50_ms"]) * 100.0, 2)
    tps_delta_pct = round(((cand["throughput_tps"] - base["throughput_tps"]) / base["throughput_tps"]) * 100.0, 2)
    mem_delta_pct = round(((cand["peak_mem_mb"] - base["peak_mem_mb"]) / base["peak_mem_mb"]) * 100.0, 2)

    rows.append({
        "device": str(device),
        "seq_len": seq_len,
        "base_p50_ms": base["p50_ms"],
        "cand_p50_ms": cand["p50_ms"],
        "latency_delta_pct": lat_delta_pct,
        "base_tps": base["throughput_tps"],
        "cand_tps": cand["throughput_tps"],
        "tps_delta_pct": tps_delta_pct,
        "base_mem_mb": base["peak_mem_mb"],
        "cand_mem_mb": cand["peak_mem_mb"],
        "mem_delta_pct": mem_delta_pct,
        "max_abs_diff": round(max_abs_diff, 5),
        "top1_match": top1_match,
        "has_nan": has_nan,
    })

csv_path = "artifacts/day48/day48-tuning-2.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print("=== DAY 48 GENERALIZATION RESULTS ===")
for r in rows:
    print(r)

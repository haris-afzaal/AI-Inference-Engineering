# Day 48: Second Bounded Tuning Experiment (Generalization Across Input Length)

## 1. Experiment Design & Hypothesis
* **Carried-Forward Tuning Variable:** `dtype` (`float32` baseline vs. `float16` candidate).
* **Varied Workload Dimension:** Input prompt sequence length (`seq_len = 128` vs. `seq_len = 512` tokens).
* **Held Constant:** `4-layer Decoder` (`d_model=1024`), `decode_steps=12`, `concurrency=1`, `warmups=2`, `reps=5`, local WSL2 CPU runtime.
* **Pre-Run Prediction:** `float16` will preserve or increase its latency speedup at `seq_len=512` while cutting weight memory by `50%` and maintaining `top-1` token equivalence.

## 2. Measured Results (`artifacts/day48/day48-tuning-2.csv`)
| `seq_len` | `float32` `p50` (ms) | `float16` `p50` (ms) | Latency $\Delta\%$ | `float32` TPS | `float16` TPS | TPS $\Delta\%$ | Memory (`MB`) | `top1_match` |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`128`** | `408.658` | `361.361` | **`-11.57%`** | `342.58` | `387.42` | **`+13.09%`** | `192.2 -> 96.1` (`-50%`) | `True` (`0 NaNs`) |
| **`512`** | `609.676` | `1953.515` | **`+220.42%`** | `859.47` | `268.23` | **`-68.79%`** | `192.2 -> 96.1` (`-50%`) | `True` (`0 NaNs`) |

## 3. Root Cause & Falsification Analysis
* **Hypothesis Falsified:** At `seq_len=128`, smaller weight bandwidth (`96.1 MB` vs. `192.2 MB`) reduced CPU cache pressure and yielded an `11.57%` speedup. At `seq_len=512` (`4x` larger prefill), execution became compute-bound; without dedicated GPU FP16 Tensor Cores, CPU fallback conversion overhead (`FP16 <-> FP32`) caused a `3.2x` (`+220.42%`) latency regression.
* **Correctness & Memory Invariants:** Weight memory stayed halved (`-50.0%`) and `top1_match` remained `True` (`max_abs_diff = 0.0813–0.1139`, `has_nan = False`) across both prompt lengths.

## 4. Generalization Verdict
* **Conclusion:** **DOES NOT GENERALIZE** across longer compute-bound sequences on CPU. A tuning optimization verified on short prompts (`128` tokens) or specific hardware (GPU Tensor Cores) must be re-validated across realistic sequence lengths and target hardware before deployment.

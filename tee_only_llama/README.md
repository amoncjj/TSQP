# TEE-Only LLaMA Benchmark

This module provides a standalone execution path that benchmarks the latency of running the entire LLaMA 3.2-1B inference inside a TEE (Trusted Execution Environment). It reuses the same model assets but bypasses the GPU-splitting logic, enabling direct comparison between split inference (TEE for non-linear ops, GPU for linear ops) and full-on-TEE execution.

## Folder Contents

- `tee_runner.py`: Entry point for orchestrating the benchmark. It loads prompts, performs generation, records latency, and writes results to a JSON file.
- `__init__.py`: Empty marker to treat this folder as a Python package.
- `benchmarks/`: Contains helper scripts for benchmarking scenarios (see below).

## Usage Overview

1. Set environment variables to control prompts, batch size, max generation length, and output location.
2. Run `tee_runner.py` inside the TEE environment (e.g., via Gramine), ensuring the model weights and tokenizer files are accessible within the enclave.
3. Collect the generated JSON output for latency comparison.

Refer to the scripts inside the `benchmarks/` directory for automation examples.

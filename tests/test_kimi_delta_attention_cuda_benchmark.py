"""CUDA benchmark; run with unittest discovery from the repository root.

KDA_BENCH_BATCH=2 KDA_BENCH_TOKENS=128 KDA_BENCH_DIM=128 KDA_BENCH_HEADS=4 \
python3 -m unittest discover -s tests -p 'test_kimi_delta_attention_cuda_benchmark.py' -v

Uses eager FP32 without autocast or compilation. Speed is reported, not asserted.
"""

import os
import statistics
import time
import unittest

import torch

from model import KimiDeltaAttention, KimiDeltaAttentionV2


@unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU is not available")
class TestKimiDeltaAttentionCUDABenchmark(unittest.TestCase):
    @staticmethod
    def positive_env(name, default):
        value = int(os.environ.get(name, default))
        if value < 1:
            raise ValueError(f"{name} must be positive")
        return value

    @staticmethod
    def benchmark(model, x, backward, warmup, runs):
        model.train(backward)

        def clear_gradients():
            model.zero_grad(set_to_none=True)
            x.grad = None

        def step():
            if backward:
                model(x).square().mean().backward()
            else:
                with torch.inference_mode():
                    model(x)

        for _ in range(warmup):
            clear_gradients()
            step()
        torch.cuda.synchronize(x.device)

        timings = []
        for _ in range(runs):
            clear_gradients()
            torch.cuda.synchronize(x.device)
            start = time.perf_counter()
            step()
            torch.cuda.synchronize(x.device)
            timings.append((time.perf_counter() - start) * 1000)

        # Separate memory measurement so allocator-stat bookkeeping is untimed.
        # Both models remain resident; subtract baseline allocations to report
        # this step's extra live tensors, including gradients for backward.
        clear_gradients()
        torch.cuda.synchronize(x.device)
        baseline = torch.cuda.memory_allocated(x.device)
        torch.cuda.reset_peak_memory_stats(x.device)
        step()
        torch.cuda.synchronize(x.device)
        peak_mib = (torch.cuda.max_memory_allocated(x.device) - baseline) / 2**20
        clear_gradients()
        return statistics.median(timings), peak_mib

    def test_v1_vs_v2(self):
        batch = self.positive_env("KDA_BENCH_BATCH", "2")
        length = self.positive_env("KDA_BENCH_TOKENS", "128")
        dim = self.positive_env("KDA_BENCH_DIM", "128")
        heads = self.positive_env("KDA_BENCH_HEADS", "4")
        warmup = self.positive_env("KDA_BENCH_WARMUP", "3")
        runs = self.positive_env("KDA_BENCH_RUNS", "10")
        self.assertEqual(dim % heads, 0, "DIM must be divisible by HEADS")

        device = torch.device("cuda", torch.cuda.current_device())
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", old_tf32)
        torch.backends.cuda.matmul.allow_tf32 = False

        # Preserve the calling test suite's CPU and CUDA RNG states.
        with torch.random.fork_rng(devices=[device.index]):
            torch.random.default_generator.manual_seed(1337)
            torch.cuda.manual_seed(1337)
            reference = KimiDeltaAttention(dim, heads).to(device)
            candidate = KimiDeltaAttentionV2(dim, heads).to(device)
            candidate.load_state_dict(reference.state_dict(), strict=True)
            x = torch.randn(batch, length, dim, device=device)

            # Confirm that the timed models compute matching outputs first.
            with torch.inference_mode():
                expected, actual = reference(x), candidate(x)
                self.assertTrue(torch.isfinite(expected).all().item())
                self.assertTrue(torch.isfinite(actual).all().item())
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
                max_error = (actual - expected).abs().max().item()
            del expected, actual

            print(
                f"\nGPU: {torch.cuda.get_device_name(device)}"
                f"\nPyTorch: {torch.__version__}; CUDA: {torch.version.cuda}"
                f"\nB={batch}, T={length}, dim={dim}, heads={heads}"
                f"\nEager FP32, TF32 off; warmup={warmup}, runs={runs}"
                f"\nMaximum output difference: {max_error:.3e}"
                "\nTiming: synchronized wall time, including Python/launch overhead"
                "\nMemory: peak allocated increase per step (not reserved memory)",
                flush=True,
            )
            for backward in (False, True):
                x.requires_grad_(backward)
                mode = "forward + backward" if backward else "inference forward"
                results = []
                for name, model in (("V1", reference), ("V2", candidate)):
                    median_ms, peak_mib = self.benchmark(model, x, backward, warmup, runs)
                    results.append(median_ms)
                    print(
                        f"  {mode:20s} {name}: {median_ms:9.3f} ms, "
                        f"{batch * length * 1000 / median_ms:10.0f} tokens/s, "
                        f"{peak_mib:9.1f} MiB extra peak",
                        flush=True,
                    )
                print(
                    f"  V2 speedup for {mode}: {results[0] / results[1]:.2f}x "
                    "(greater than 1 means V2 is faster)",
                    flush=True,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)

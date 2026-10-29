import os
import statistics
import time
import unittest

import torch
import torch.nn.functional as F


class TestMPSLinearBenchmark(unittest.TestCase):
    """Compare FP32 and BF16 F.linear performance on Apple MPS."""

    batch_tokens = int(os.environ.get("LINEAR_BENCH_TOKENS", "4096"))
    in_features = int(os.environ.get("LINEAR_BENCH_IN", "384"))
    out_features = int(os.environ.get("LINEAR_BENCH_OUT", "1536"))
    warmup_runs = int(os.environ.get("LINEAR_BENCH_WARMUP", "5"))
    measured_runs = int(os.environ.get("LINEAR_BENCH_RUNS", "20"))

    @staticmethod
    def _benchmark(x, weight, warmup_runs, measured_runs):
        for _ in range(warmup_runs):
            F.linear(x, weight)
        torch.mps.synchronize()

        timings_ms = []
        for _ in range(measured_runs):
            start = time.perf_counter()
            F.linear(x, weight)
            torch.mps.synchronize()
            timings_ms.append((time.perf_counter() - start) * 1_000)

        return timings_ms

    @unittest.skipUnless(
        torch.backends.mps.is_available(),
        "MPS is not available in this process",
    )
    def test_fp32_vs_bf16_linear(self):
        torch.manual_seed(1337)
        device = torch.device("mps")

        x_fp32 = torch.randn(
            self.batch_tokens,
            self.in_features,
            device=device,
            dtype=torch.float32,
        )
        weight_fp32 = torch.randn(
            self.out_features,
            self.in_features,
            device=device,
            dtype=torch.float32,
        )
        x_bf16 = x_fp32.to(torch.bfloat16)
        weight_bf16 = weight_fp32.to(torch.bfloat16)

        output_fp32 = F.linear(x_fp32, weight_fp32)
        output_bf16 = F.linear(x_bf16, weight_bf16)
        torch.mps.synchronize()

        self.assertEqual(output_fp32.dtype, torch.float32)
        self.assertEqual(output_bf16.dtype, torch.bfloat16)
        torch.testing.assert_close(
            output_bf16.float(),
            output_fp32,
            rtol=5e-2,
            atol=5e-1,
        )

        fp32_times = self._benchmark(
            x_fp32,
            weight_fp32,
            self.warmup_runs,
            self.measured_runs,
        )
        bf16_times = self._benchmark(
            x_bf16,
            weight_bf16,
            self.warmup_runs,
            self.measured_runs,
        )

        fp32_median = statistics.median(fp32_times)
        bf16_median = statistics.median(bf16_times)
        speedup = fp32_median / bf16_median

        print(
            "\nMPS F.linear benchmark"
            f"\n  shape: ({self.batch_tokens}, {self.in_features}) @ "
            f"({self.out_features}, {self.in_features}).T"
            f"\n  FP32 median: {fp32_median:.3f} ms"
            f"\n  BF16 median: {bf16_median:.3f} ms"
            f"\n  BF16 speedup: {speedup:.3f}x"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

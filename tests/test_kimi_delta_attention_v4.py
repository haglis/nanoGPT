import unittest

import torch

from model import GPT, GPTConfig, KimiDeltaAttention, KimiDeltaAttentionV4


class TestKimiDeltaAttentionV4(unittest.TestCase):
    def test_cpu_fallback_outputs_and_gradients(self):
        with torch.random.fork_rng():
            torch.manual_seed(42)
            reference = KimiDeltaAttention(8, 2).double()
            candidate = KimiDeltaAttentionV4(8, 2, 32).double()
            candidate.load_state_dict(reference.state_dict(), strict=True)
            x = torch.randn(2, 35, 8, dtype=torch.float64, requires_grad=True)
            y = x.detach().clone().requires_grad_(True)
            expected, actual = reference(x), candidate(y)
            torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-12)
            upstream = torch.randn_like(expected)
            expected.backward(upstream)
            actual.backward(upstream)
            torch.testing.assert_close(y.grad, x.grad, rtol=1e-9, atol=1e-12)
            for name, parameter in candidate.named_parameters():
                torch.testing.assert_close(
                    parameter.grad, dict(reference.named_parameters())[name].grad,
                    rtol=1e-9, atol=1e-12,
                )

    def test_gpt_uses_v4_and_trains_on_cpu(self):
        with torch.random.fork_rng():
            torch.manual_seed(42)
            model = GPT(GPTConfig(n_layer=1, n_head=2, n_embd=8,
                                  vocab_size=16, block_size=8, chunk_size=32))
            self.assertIsInstance(model.transformer.h[0].attn, KimiDeltaAttentionV4)
            tokens = torch.randint(16, (2, 8))
            logits, loss = model(tokens, tokens)
            self.assertEqual(logits.shape, (2, 8, 16))
            self.assertTrue(torch.isfinite(loss).item())
            loss.backward()

    def test_rejects_unsupported_chunk_size(self):
        with self.assertRaisesRegex(ValueError, "32 or 64"):
            KimiDeltaAttentionV4(8, 2, 16)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for official FLA kernel")
    def test_official_cuda_outputs_and_gradients(self):
        # Missing FLA on a CUDA machine must fail, not silently skip/fall back.
        from fla.ops.kda import chunk_kda  # noqa: F401

        device = torch.cuda.current_device()
        with torch.random.fork_rng(devices=[device]):
            torch.random.default_generator.manual_seed(42)
            torch.cuda.manual_seed(42)
            for chunk_size in (32, 64):
                for length in (1, 65, 128):
                    with self.subTest(chunk_size=chunk_size, length=length):
                        reference = KimiDeltaAttention(128, 2).cuda()
                        candidate = KimiDeltaAttentionV4(128, 2, chunk_size).cuda()
                        candidate.load_state_dict(reference.state_dict(), strict=True)
                        x = torch.randn(2, length, 128, device='cuda', requires_grad=True)
                        y = x.detach().clone().requires_grad_(True)
                        expected, actual = reference(x), candidate(y)
                        self.assertEqual(actual.shape, expected.shape)
                        self.assertTrue(torch.isfinite(actual).all().item())
                        # FLA uses reduced-precision intermediate matmuls.
                        self.assert_relative_error(actual, expected)
                        upstream = torch.randn_like(expected)
                        expected.backward(upstream)
                        actual.backward(upstream)
                        self.assert_relative_error(y.grad, x.grad)
                        for name, parameter in candidate.named_parameters():
                            with self.subTest(parameter=name):
                                self.assert_relative_error(
                                    parameter.grad,
                                    dict(reference.named_parameters())[name].grad,
                                )

    def assert_relative_error(self, actual, expected):
        self.assertIsNotNone(actual)
        self.assertTrue(torch.isfinite(actual).all().item())
        relative = (actual - expected).float().norm() / expected.float().norm().clamp_min(1e-12)
        self.assertLess(relative.item(), 0.02)


if __name__ == '__main__':
    unittest.main()

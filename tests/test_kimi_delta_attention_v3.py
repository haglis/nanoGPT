"""V3 correctness against the original token-by-token recurrence (V1)."""

import unittest

import torch

from model import KimiDeltaAttention, KimiDeltaAttentionV3


class TestKimiDeltaAttentionV3(unittest.TestCase):
    def setUp(self):
        rng_state = torch.random.get_rng_state()
        self.addCleanup(torch.random.set_rng_state, rng_state)
        torch.manual_seed(1234)

    @staticmethod
    def make_models(dim, heads, dtype, chunk_size):
        reference = KimiDeltaAttention(dim, heads).to(dtype=dtype)
        with torch.no_grad():
            for parameter in reference.parameters():
                parameter.uniform_(-0.3, 0.3)
        candidate = KimiDeltaAttentionV3(dim, heads).to(dtype=dtype)
        candidate.chunk_size = chunk_size
        candidate.load_state_dict(reference.state_dict(), strict=True)
        return reference, candidate

    def assert_output(self, actual, expected, dtype):
        self.assertIsInstance(actual, torch.Tensor, "V3.forward() must return a tensor")
        self.assertTrue(torch.isfinite(actual).all().item())
        rtol, atol = (1e-10, 1e-12) if dtype == torch.float64 else (1e-4, 1e-6)
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)

    def test_outputs_match_reference_for_chunk_sizes(self):
        # Single token, multiple batches/heads, head_dim=1, and longer sequence.
        cases = [(1, 1, 4, 1), (2, 7, 6, 2), (3, 6, 4, 4), (2, 17, 8, 2)]
        for dtype in (torch.float32, torch.float64):
            for batch, length, dim, heads in cases:
                # One-token chunks, exact/partial chunks, full and oversized chunks.
                for chunk_size in sorted({1, 2, 3, length, length + 3}):
                    with self.subTest(dtype=dtype, shape=(batch, length, dim),
                                      heads=heads, chunk_size=chunk_size):
                        reference, candidate = self.make_models(dim, heads, dtype, chunk_size)
                        x = torch.randn(batch, length, dim, dtype=dtype)
                        with torch.no_grad():
                            self.assert_output(candidate(x), reference(x), dtype)

    def test_noncontiguous_input_matches_reference(self):
        reference, candidate = self.make_models(6, 2, torch.float64, 3)
        x = torch.randn(2, 6, 7, dtype=torch.float64).transpose(1, 2)
        self.assertFalse(x.is_contiguous())
        with torch.no_grad():
            self.assert_output(candidate(x), reference(x), x.dtype)

    def test_hand_calculated_update_across_chunk_boundary(self):
        reference, candidate = self.make_models(2, 1, torch.float64, 1)
        q = torch.tensor([[2., 1.], [1., -1.]], dtype=torch.float64)
        k = torch.tensor([[1., 2.], [2., 1.]], dtype=torch.float64)
        v = torch.tensor([[3., 1.], [1., 4.]], dtype=torch.float64)
        with torch.no_grad():
            reference.qkv.layer.weight.copy_(torch.cat([q, k, v], dim=1).T)
            reference.beta.layer.weight.zero_()
            reference.alpha_fc1.layer.weight.zero_()
            reference.alpha_fc2.layer.weight.zero_()
            reference.alpha_fc2.layer.bias.copy_(
                torch.logit(torch.tensor([0.25, 0.75], dtype=torch.float64))
            )
            candidate.load_state_dict(reference.state_dict())
            x = torch.eye(2, dtype=torch.float64).unsqueeze(0)
            # S1=[[1.5,.5],[3,1]], o1=[6,2]. Decay S1, then correct
            # v2 by k2 @ decayed_S1=[3,1]: delta2=[-2,3].
            # S2=[[-1.625,3.125],[1.25,2.25]], o2=[-2.875,.875].
            expected = torch.tensor([[[6., 2.], [-2.875, .875]]], dtype=x.dtype)
            self.assert_output(reference(x), expected, x.dtype)
            self.assert_output(candidate(x), expected, x.dtype)

    def test_gradients_cross_chunk_boundaries(self):
        for dtype in (torch.float32, torch.float64):
            for chunk_size in (1, 3, 7, 10):
                with self.subTest(dtype=dtype, chunk_size=chunk_size):
                    reference, candidate = self.make_models(6, 2, dtype, chunk_size)
                    x_ref = torch.randn(2, 7, 6, dtype=dtype, requires_grad=True)
                    x_v3 = x_ref.detach().clone().requires_grad_(True)
                    expected, actual = reference(x_ref), candidate(x_v3)
                    self.assert_output(actual, expected, dtype)
                    # Only the final token contributes to the loss. Earlier
                    # input gradients must flow through the boundary states.
                    upstream = torch.zeros_like(expected)
                    upstream[:, -1] = torch.randn_like(expected[:, -1])
                    expected.backward(upstream)
                    actual.backward(upstream)
                    self.assertGreater(x_ref.grad[:, :1].abs().sum().item(), 0)
                    rtol, atol = (1e-9, 1e-11) if dtype == torch.float64 else (1e-4, 1e-6)
                    torch.testing.assert_close(x_v3.grad, x_ref.grad, rtol=rtol, atol=atol)
                    ref_parameters = dict(reference.named_parameters())
                    for name, parameter in candidate.named_parameters():
                        with self.subTest(parameter=name):
                            self.assertIsNotNone(parameter.grad)
                            self.assertIsNotNone(ref_parameters[name].grad)
                            torch.testing.assert_close(
                                parameter.grad, ref_parameters[name].grad,
                                rtol=rtol, atol=atol,
                            )

    def test_repeated_calls_reset_state(self):
        reference, candidate = self.make_models(6, 2, torch.float64, 3)
        x = torch.randn(2, 7, 6, dtype=torch.float64)
        with torch.no_grad():
            expected = reference(x)
            self.assert_output(candidate(x), expected, x.dtype)
            candidate(torch.randn_like(x))
            self.assert_output(candidate(x), expected, x.dtype)


if __name__ == "__main__":
    unittest.main()

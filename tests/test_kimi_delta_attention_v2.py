import unittest

import torch

from model import KimiDeltaAttention, KimiDeltaAttentionV2


class TestKimiDeltaAttentionV2(unittest.TestCase):
    def setUp(self):
        rng_state = torch.random.get_rng_state()
        self.addCleanup(torch.random.set_rng_state, rng_state)
        torch.manual_seed(1234)

    @staticmethod
    def make_models(dim, heads, dtype):
        reference = KimiDeltaAttention(dim, heads).to(dtype=dtype)
        # Exercise appreciable recurrent corrections and varying gates, rather
        # than only the tiny outputs produced by the default initialization.
        with torch.no_grad():
            for parameter in reference.parameters():
                parameter.uniform_(-0.3, 0.3)
        candidate = KimiDeltaAttentionV2(dim, heads).to(dtype=dtype)
        candidate.load_state_dict(reference.state_dict(), strict=True)
        return reference, candidate

    def test_forward_output_shape_matches_original(self):
        reference, candidate = self.make_models(6, 2, torch.float64)
        x = torch.randn(2, 5, 6, dtype=torch.float64)
        self.assertEqual(candidate(x).shape, reference(x).shape)

    def test_forward_values_match_original(self):
        cases = [(1, 1, 4, 1), (2, 5, 6, 2), (3, 7, 4, 4), (2, 32, 8, 2)]
        for dtype in (torch.float64, torch.float32):
            rtol, atol = (1e-10, 1e-12) if dtype == torch.float64 else (1e-4, 1e-6)
            for batch, length, dim, heads in cases:
                for noncontiguous in (False, True):
                    with self.subTest(dtype=dtype, shape=(batch, length, dim),
                                      heads=heads, noncontiguous=noncontiguous):
                        reference, candidate = self.make_models(dim, heads, dtype)
                        if noncontiguous:
                            x = torch.randn(batch, dim, length, dtype=dtype).transpose(1, 2)
                        else:
                            x = torch.randn(batch, length, dim, dtype=dtype)
                        with torch.no_grad():
                            expected = reference(x)
                            actual = candidate(x)
                        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)

    def test_input_and_parameter_gradients_match_original(self):
        for dtype in (torch.float64, torch.float32):
            with self.subTest(dtype=dtype):
                rtol, atol = (1e-9, 1e-11) if dtype == torch.float64 else (1e-4, 1e-6)
                reference, candidate = self.make_models(6, 2, dtype)
                x_reference = torch.randn(2, 7, 6, dtype=dtype, requires_grad=True)
                x_candidate = x_reference.detach().clone().requires_grad_(True)
                expected = reference(x_reference)
                actual = candidate(x_candidate)
                # The same arbitrary upstream gradient checks each output's
                # contribution, without the symmetry of a simple output sum.
                upstream = torch.randn_like(expected)
                expected.backward(upstream)
                actual.backward(upstream)
                torch.testing.assert_close(
                    x_candidate.grad, x_reference.grad, rtol=rtol, atol=atol
                )
                reference_parameters = dict(reference.named_parameters())
                for name, parameter in candidate.named_parameters():
                    with self.subTest(parameter=name):
                        self.assertIsNotNone(parameter.grad)
                        self.assertIsNotNone(reference_parameters[name].grad)
                        torch.testing.assert_close(
                            parameter.grad, reference_parameters[name].grad,
                            rtol=rtol, atol=atol,
                        )


if __name__ == "__main__":
    unittest.main()

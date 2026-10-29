import unittest

import torch

from model import KimiDeltaAttention


class TestKimiDeltaAttention(unittest.TestCase):
    def setUp(self):
        self.rng_state = torch.random.get_rng_state()
        self.addCleanup(torch.random.set_rng_state, self.rng_state)
        torch.manual_seed(1234)

    def test_shape_dtype_and_finite_output(self):
        for batch, length, dim, heads in [(1, 1, 4, 1), (2, 5, 8, 2), (3, 2, 4, 4)]:
            for dtype in (torch.float32, torch.float64):
                with self.subTest(shape=(batch, length, dim), heads=heads, dtype=dtype):
                    model = KimiDeltaAttention(dim, heads).to(dtype=dtype)
                    x = torch.randn(batch, length, dim, dtype=dtype)
                    output = model(x)
                    self.assertEqual(output.shape, x.shape)
                    self.assertEqual(output.dtype, x.dtype)
                    self.assertEqual(output.device, x.device)
                    self.assertTrue(torch.isfinite(output).all().item())

    def test_two_token_delta_update(self):
        """Check decay, error correction, and readout against hand calculations."""
        model = KimiDeltaAttention(2, 1).double()
        with torch.no_grad():
            # q = k = v = x; beta = 1/2; alpha = [1/4, 3/4].
            model.qkv.layer.weight.copy_(torch.eye(2).repeat(3, 1))
            model.beta.layer.weight.zero_()
            model.alpha_fc1.layer.weight.zero_()
            model.alpha_fc2.layer.weight.zero_()
            model.alpha_fc2.layer.bias.copy_(torch.logit(torch.tensor([0.25, 0.75])))

        x = torch.tensor([[[1.0, 2.0], [2.0, 1.0]]], dtype=torch.float64)
        # S1 = [[.5, 1], [1, 2]], giving q1 @ S1 = [2.5, 5].
        # Decayed S1 = [[.125, .25], [.75, 1.5]].
        # Prediction = [1, 2], delta = [1, -1].
        # S2 = [[1.125, -.75], [1.25, 1]], giving [3.5, -.5].
        expected = torch.tensor([[[2.5, 5.0], [3.5, -0.5]]], dtype=torch.float64)
        torch.testing.assert_close(model(x), expected, rtol=1e-6, atol=1e-7)

    def test_explicit_qkv_matches_hand_calculated_output(self):
        """Two tokens, one head: distinct 2 x 2 Q, K, V (rows are tokens)."""
        model = KimiDeltaAttention(2, 1).double()
        q = torch.tensor([[2., 1.], [1., -1.]], dtype=torch.float64)
        k = torch.tensor([[1., 2.], [2., 1.]], dtype=torch.float64)
        v = torch.tensor([[3., 1.], [1., 4.]], dtype=torch.float64)
        x = torch.eye(2, dtype=torch.float64).unsqueeze(0)

        with torch.no_grad():
            # Linear(x) = x @ weight.T. Identity input gives exactly Q, K, V.
            model.qkv.layer.weight.copy_(torch.cat([q, k, v], dim=1).T)
            model.beta.layer.weight.zero_()  # sigmoid(0) = 1/2
            model.alpha_fc1.layer.weight.zero_()
            model.alpha_fc2.layer.weight.zero_()
            model.alpha_fc2.layer.bias.copy_(
                torch.logit(torch.tensor([0.25, 0.75], dtype=torch.float64))
            )
            torch.testing.assert_close(model.qkv(x)[0], torch.cat([q, k, v], dim=1))

            # Row-vector notation; S0 = 0, alpha = [1/4, 3/4], beta = 1/2:
            # S_bar = diag(alpha) @ S_previous
            # delta = v_t - k_t @ S_bar
            # S_t = S_bar + beta * outer(k_t, delta); o_t = q_t @ S_t.
            #
            # Token 1: S_bar = 0, delta = [3, 1].
            # S1 = 1/2 * outer([1, 2], [3, 1]) = [[1.5, .5], [3, 1]].
            # o1 = [2, 1] @ S1 = [6, 2].
            #
            # Token 2: S_bar = [[.375, .125], [2.25, .75]].
            # Prediction = [2, 1] @ S_bar = [3, 1].
            # delta = [1, 4] - [3, 1] = [-2, 3].
            # Update = 1/2 * outer([2, 1], [-2, 3]) = [[-2, 3], [-1, 1.5]].
            # S2 = [[-1.625, 3.125], [1.25, 2.25]].
            # o2 = [1, -1] @ S2 = [-2.875, .875].
            expected = torch.tensor(
                [[[6., 2.], [-2.875, 0.875]]], dtype=torch.float64
            )
            actual = model(x)

        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)

    def test_future_tokens_do_not_change_prefix(self):
        model = KimiDeltaAttention(8, 2)
        x = torch.randn(2, 6, 8)
        changed = x.clone()
        changed[:, 3:] = torch.randn_like(changed[:, 3:]) * 5
        with torch.no_grad():
            output = model(x)
            torch.testing.assert_close(output[:, :3], model(changed)[:, :3])
            torch.testing.assert_close(output[:, :3], model(x[:, :3]))

    def test_batches_are_independent_and_state_resets(self):
        model = KimiDeltaAttention(8, 2)
        x = torch.randn(3, 4, 8)
        with torch.no_grad():
            batched = model(x)
            separate = torch.cat([model(sample.unsqueeze(0)) for sample in x])
            torch.testing.assert_close(batched, separate)
            model(torch.randn_like(x))
            torch.testing.assert_close(model(x), batched)

    def test_noncontiguous_input(self):
        model = KimiDeltaAttention(8, 2)
        x = torch.randn(2, 8, 5).transpose(1, 2)
        self.assertFalse(x.is_contiguous())
        torch.testing.assert_close(model(x), model(x.contiguous()))

    def test_zero_input_produces_zero_output(self):
        model = KimiDeltaAttention(8, 2)
        x = torch.zeros(2, 4, 8)
        torch.testing.assert_close(model(x), x, rtol=0, atol=0)

    def test_backward_reaches_input_and_all_parameters(self):
        model = KimiDeltaAttention(4, 2).double()
        x = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
        model(x).square().sum().backward()
        for name, tensor in [("input", x), *model.named_parameters()]:
            with self.subTest(tensor=name):
                self.assertIsNotNone(tensor.grad)
                self.assertTrue(torch.isfinite(tensor.grad).all().item())
                self.assertGreater(tensor.grad.abs().sum().item(), 0)

    def test_input_gradient_matches_finite_differences(self):
        model = KimiDeltaAttention(4, 2).double()
        # Larger weights keep the numerical check sensitive to recurrent terms.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.uniform_(-0.5, 0.5)
        x = torch.randn(1, 3, 4, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(model, (x,)))

    def test_rejects_dimension_not_divisible_by_heads(self):
        with self.assertRaises(AssertionError):
            KimiDeltaAttention(7, 2)


if __name__ == "__main__":
    unittest.main()

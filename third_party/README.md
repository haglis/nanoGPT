# Official KDA source

`flash-linear-attention.tar.gz` is an unmodified `git archive` of
https://github.com/fla-org/flash-linear-attention at commit
`8024667ab58fdd8986587147fca71fecc017f977`, downloaded on 2026-10-05.
It includes the upstream MIT license. Moonshot's Kimi Linear repository
links to this implementation: https://github.com/MoonshotAI/Kimi-Linear.

Install from the nanoGPT root on the CUDA machine:

```bash
python -m pip install -r requirements-kda.txt
python train.py config/train_shakespeare_char.py --device=cuda --compile=False
```

Start without torch.compile to separate Triton kernel compilation from model
compilation. Warm up before measuring performance.

`KimiDeltaAttentionV4` is selected by `Block`. It wraps `fla.ops.kda.chunk_kda`
with the existing learned projections, sigmoid beta, log-sigmoid decay, and
scale=1. Query/key normalization and FLA's alternate gate activation remain
disabled to preserve the V1-V3 recurrence and checkpoint parameter names.
This integrates the official attention kernel, not the full official Kimi layer
(which has additional architectural components).

CUDA always uses FLA; missing dependencies raise an installation error.
CPU/MPS explicitly use V3, so local tests and sampling do not benchmark FLA.
The official kernel supports chunk sizes 32 and 64 and key head dimensions
up to 256. The first CUDA call compiles Triton kernels. Mixed precision may
produce small numerical differences from the FP32 reference.

GPU tests:

```bash
python -m unittest discover -s tests -p 'test_kimi_delta_attention_v4.py' -v
```

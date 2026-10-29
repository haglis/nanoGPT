import os

import torch
class Config:
    ndim = 384 #384
    batch_size = 32
    gradient_accumulation_steps = 2
    context_length = 256             
    n_layers = 8
    device = 'cuda' if torch.cuda.is_available() else 'mps' #'cpu' #'cpu' #'mps'
    n_heads = 6
    lr = 3e-4
    weight_decay = 0.1
    max_steps = 5000
    steps_to_eval = 10
    voc_size = 65
    compile = False
    # bf16 keeps the existing custom forward/backward baseline unchanged.
    # mxfp8 uses Blackwell MXFP8 block-scaled training through TorchAO.
    linear_mode = os.environ.get("LINEAR_MODE", "bf16")

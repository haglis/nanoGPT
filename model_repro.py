import torch.nn as nn
import math, torch
from config import Config

class Norm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        #self.norm = nn.RMSNorm(dim)
        #self.norm.weight.requires_grad_ = False

    def forward(self, x):
        return nn.functional.rms_norm(x,(x.size(-1),))

class LinearLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer = nn.Linear(self.in_dim, self.out_dim, bias = bias)
        nn.init.normal_(self.layer.weight, mean=0, std=0.02)#math.sqrt(6.0/(self.in_dim + self.out_dim)))
        if bias:
            nn.init.zeros_(self.layer.bias)

    def forward(self, x):
        return self.layer(x)
    








class MHA(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        assert(dim % n_heads == 0)
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads 
        self.qkv = LinearLayer(self.dim, 3*self.dim)
        self.proj = LinearLayer(self.dim, self.dim)
        self.register_buffer("bias",torch.tril(torch.ones(1024, 1024)).view(1,1,1024,1024))

    def forward(self, x) -> torch.tensor:
        B, T, D = x.size()

        q, k, v = torch.split(self.qkv(x), self.dim, dim=-1)
       # import pdb; pdb.set_trace()

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1,2) # B, n_heads, T, head_dim
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1,2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1,2)
        

        mat = q @ k.transpose(-1, -2) * 1.0 / math.sqrt(k.size(-1))

        mat = mat.masked_fill(self.bias[:,:,:T, :T]==0, float('-inf'))

        mat = nn.functional.softmax(mat, dim = -1)

        y = mat @ v 

        y = y.transpose(1,2).contiguous().view(B, T, self.dim)

        return self.proj(y)

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.act = nn.ReLU()
        self.dim = dim
        self.fc1 = LinearLayer(self.dim, 4 * self.dim)
        self.fc2 = LinearLayer(self.dim * 4, self.dim)

    def forward(self, x) -> torch.tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        return x 

class Transformer(nn.Module):

    def __init__(self, dim: int, nheads: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.nheads = nheads

        self.mlp = MLP(self.dim)
        self.mha = MHA(self.dim, self.nheads)

        self.norm1 = Norm(self.dim)
        self.norm2 = Norm(self.dim)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = x + self.mha(self.norm1(x))

        x = x + self.mlp(self.norm2(x))

        return x

class Model(nn.Module):

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_length = config.context_length
        self.webd = nn.Embedding(config.voc_size, config.ndim)
        self.pebd = nn.Embedding(config.context_length, config.ndim)
        nn.init.normal_(self.pebd.weight,mean=0,std=0.02)#math.sqrt(6.0/(config.context_length+config.ndim)))
        nn.init.normal_(self.webd.weight,mean=0,std=0.02)#math.sqrt(6.0/(config.voc_size+config.ndim)))

        self.transformers = nn.ModuleList([Transformer(config.ndim, config.n_heads) for _ in range(config.n_layers)])
        self.lm_head = LinearLayer(config.ndim, config.voc_size)
        self.lm_head.layer.weight = self.webd.weight
        

    def forward(self, idx, targets=None) -> torch.tensor:
        #import pdb;pdb.set_trace()
        x = self.webd(idx)
        pos = torch.arange(0, self.context_length, device=idx.device)
        x = x + self.pebd(pos)

        for transformer in self.transformers:
            x = transformer(x)

        logits = self.lm_head(x)
        if targets != None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return loss
        else: return logits
import torch
import torch.nn as nn
from config import Config

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        #self.norm = nn.LayerNorm(dim)
        self.dim = dim
    def forward(self, x):
        return nn.functional.rms_norm(x,(x.size(-1), ))
        #return self.norm(x)

class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim, linear_mode="bf16"):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear_mode = linear_mode
        self.layer = nn.Linear(in_dim, out_dim, bias=False)
        torch.nn.init.normal_(self.layer.weight, mean=0, std=0.02)

    def forward(self, x: torch.tensor):
        #assert x.dtype == torch.bfloat16
        #x = x.to(torch.bfloat16)
        #weight = self.layer.weight.to(torch.bfloat16)
        #import pdb;pdb.set_trace()
        #return  x @ weight.t()
        
        return self.layer(x)
        #return nn.functional.linear(x, weight)
        if self.linear_mode == "bf16":
            return CustomLinearFunction.apply(x, self.layer.weight, self.layer.bias)
        if self.linear_mode == "mxfp8":
            # apply_mxfp8_training wraps this parameter with TorchAO's MXFP8
            # training tensor. F.linear then dispatches to its custom MXFP8
            # forward/backward implementation.
            return self.layer(x)
        raise ValueError(f"unsupported linear mode: {self.linear_mode}")
    
    
class CustomLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        input_bf16_2d = input.view(-1,input.size(-1)).to(torch.bfloat16)
        weight_bf16 = weight.to(torch.bfloat16)
        ctx.save_for_backward(input_bf16_2d, weight_bf16)
        out_2d = torch.mm(input_bf16_2d,
                               weight_bf16.t())
        out = out_2d.view(input.size(0), -1, weight.size(0))
        return out 
    
        precision=torch.float8_e4m3fn
        max_w = weight.abs().max().float()
        max_x = input.abs().max().float()
        FLOAT8_e4m3fn_MAX = torch.finfo(precision).max
        FLOAT8_e4m3fn_MIN = torch.finfo(precision).min
        scale_x = FLOAT8_e4m3fn_MAX / max_x 
        scale_w = FLOAT8_e4m3fn_MAX / max_w
        x = (input * scale_x).clamp(FLOAT8_e4m3fn_MIN, FLOAT8_e4m3fn_MAX).to(torch.float8_e4m3fn)
        weight = (weight * scale_w).clamp(FLOAT8_e4m3fn_MIN, FLOAT8_e4m3fn_MAX).to(torch.float8_e4m3fn)
                #import pdb;pdb.set_trace()
        ctx.save_for_backward(x, weight, bias, scale_x, scale_w)
        #import pdb; pdb.set_trace()
        out = torch._scaled_mm(
                x.view(-1, x.size(-1)),
                weight.t(),
                scale_a=1.0 / scale_x,
                scale_b=1.0 / scale_w,
                out_dtype=input.dtype,
                use_fast_accum=True,
                )
        #out =  x @ weight.t()
        #out = (out / scale_x / scale_w ).clamp(FLOAT8_e4m3fn_MIN, FLOAT8_e4m3fn_MAX)
        out = out.view(x.size(0), -1, weight.size(0)) #.to(torch.bfloat16)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        #input, weight, bias = ctx.saved_tensors
        """A standard Linear layer does one matmul in forward and two in backward:
            forward:      output     = input      @ weight.T
            backward:     grad_input = grad_output @ weight
                            grad_weight= grad_output.T @ input
        """
        #import pdb;pdb.set_trace()
        input_bf16_2d, weight_bf16 = ctx.saved_tensors
        grad_output_2d = grad_output.view(-1, grad_output.size(-1))
        grad_input_2d = torch.mm(grad_output_2d,
                                         weight_bf16.t().contiguous().t())
        grad_input = grad_input_2d.view(grad_output.size(0), -1, grad_input_2d.size(-1))
        grad_weight = torch.mm(grad_output_2d.t().contiguous(),
                                       input_bf16_2d.t().contiguous().t()
                                       )
        return grad_input, grad_weight, None

        input_fp8, weight_fp8, bias_fp8, scale_x, scale_w = ctx.saved_tensors
        precision=torch.float8_e4m3fn
        #grad_output_pre = grad_output.to(precision)
        #print(grad_output_float32.dtype, weight.dtype)
        
        max_g = grad_output.abs().max().float()
        precision = torch.float8_e4m3fn
        FLOAT8_e4m3fn_MAX = torch.finfo(precision).max
        FLOAT8_e4m3fn_MIN = torch.finfo(precision).min
        scale_g = FLOAT8_e4m3fn_MAX / max_g
        grad_output_fp8 = (grad_output * scale_g).clamp(FLOAT8_e4m3fn_MIN, FLOAT8_e4m3fn_MAX).to(precision)
        grad_output_2d = grad_output_fp8.view(-1, grad_output.size(-1))
        #import pdb; pdb.set_trace()
        grad_input_2d = torch._scaled_mm(
            grad_output_2d,
            weight_fp8.t().contiguous().t(),
            scale_a = 1.0 / scale_g,
            scale_b = 1.0 / scale_w,
            out_dtype = grad_output.dtype
        )
        grad_input = grad_input_2d.view(grad_output.size(0), -1, grad_input_2d.size(-1))
        grad_weight_fp32 = torch._scaled_mm(
            grad_output_fp8.view(-1, grad_output_fp8.size(-1)).transpose(-1,-2).contiguous(),
            input_fp8.view(-1, input_fp8.size(-1)).t().contiguous().t(),
            scale_a = 1.0 / scale_g,
            scale_b = 1.0 / scale_x,
            out_dtype = torch.float32
        )
        #grad_input = grad_input_2d.view(grad_output.size(0), -1, weight.size(-1))
        #grad_weight = grad_output_2d.transpose(-1,-2) @ input.view(-1,input.size(-1))
        #grad_weight_fp32 = grad_weight.to(torch.float32)
        return grad_input, grad_weight_fp32, None

import math
batch_size, context_length = 256, 1024
class Multi_attn(nn.Module):
    def __init__(self, in_dim, dim, linear_mode="bf16"):
        super().__init__()
        self.dim = dim
        assert in_dim % dim == 0
        self.n_heads = in_dim // dim 
        self.in_dim = in_dim 
        self.attn = LinearLayer(self.in_dim, 3 * self.in_dim, linear_mode)
        self.proj = LinearLayer(self.in_dim, self.in_dim, linear_mode)
        self.flash = False
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            self.flash=True
        if not self.flash:
            self.register_buffer("mask", torch.tril(torch.ones((1, 1, context_length, context_length))) )

    def forward(self, x):
        B, C, D = x.size() # batch_size, Context_length, feature dimension
        
        k, q, v = torch.split(self.attn(x), self.in_dim, dim=-1) #.split(self.in_dim)

        k = k.view(B, C, self.n_heads, self.dim).transpose(1, 2)
        q = q.view(B, C, self.n_heads, self.dim).transpose(1, 2)
        v = v.view(B, C, self.n_heads, self.dim).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal = True)
        else:
            mat = q @ k.transpose(-1,-2) * 1.0 / math.sqrt(k.size(-1))
            mat.masked_fill_(self.mask[:, :, :C, :C]==0, float('-inf'))
            attn = torch.nn.functional.softmax(mat, dim=-1)
            y = attn @ v 
        y = y.transpose(1,2).contiguous().view(B, C, D)
        y = self.proj(y)
        return y 

class MLP(nn.Module):
    def __init__(self, in_dim, linear_mode="bf16"):
        super().__init__()
        self.dim = 4 * in_dim
        self.in_proj = LinearLayer(in_dim, self.dim, linear_mode)
        self.out_proj = LinearLayer(self.dim, in_dim, linear_mode)
        #nn.init.zeros_(self.in_proj.layer.weight)
        self.gelu = nn.SiLU()
    def forward(self, x):
        x = self.in_proj(x)
        x = self.gelu(x)
        x = self.out_proj(x)
        return x

class Transformer(nn.Module):
    def __init__(self, dim, n_heads, linear_mode="bf16"):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim  
        self.attn = Multi_attn(self.dim, self.dim // n_heads, linear_mode)
        self.mlp = MLP(self.dim, linear_mode)
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
class Model(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.transformers = nn.ModuleList(
            [Transformer(config.ndim,config.n_heads,config.linear_mode) for i in range(config.n_layers)])
        self.webd = nn.Embedding(config.voc_size, config.ndim)
        torch.nn.init.normal_(self.webd.weight,mean=0,std=0.02)
        self.pebd = nn.Embedding(config.context_length, config.ndim)
        torch.nn.init.normal_(self.pebd.weight,mean=0,std=0.02)
        self.lm_head = nn.Linear(self.config.ndim,config.voc_size, bias=False)
        self.lm_head.weight = self.webd.weight
        #nn.init.zeros_(self.lm_head.weight)
        #self.loss = nn.CrossEntropyLoss()

    def forward(self, idx, y=None):
        #import pdb;pdb.set_trace()
        webd = self.webd(idx)
        pebd = self.pebd(torch.arange(0,self.config.context_length, device=self.config.device))
        x = webd + pebd
        for i in range(self.config.n_layers):
            x = self.transformers[i](x)
        x = self.lm_head(x)
        #import pdb;pdb.set_trace()
        if y==None:
            return x 
        else: 
            loss = torch.nn.functional.cross_entropy(x.transpose(-1,-2), y) #self.loss(x, y)
            return loss 
        
         

        

def apply_mxfp8_training(model):
    """Enable native MXFP8 forward and backward GEMMs on eligible linears.

    MXFP8 uses one E8M0 power-of-two scale per block of 32 FP8 values. TorchAO
    owns the layout-specific casting kernels and the custom autograd function;
    keeping them visible to torch.compile is essential for performance.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("MXFP8 requires a CUDA device")
    major, _minor = torch.cuda.get_device_capability()
    if major < 10:
        raise RuntimeError("MXFP8 requires NVIDIA Blackwell (SM100 or newer)")

    try:
        from torchao.quantization import quantize_
        from torchao.prototype.moe_training.config import (
            MXFP8TrainingOpConfig,
            MXFP8TrainingRecipe,
        )
    except ImportError as exc:
        raise RuntimeError(
            "MXFP8 training requires a recent compatible PyTorch and TorchAO"
        ) from exc

    mx_config = MXFP8TrainingOpConfig.from_recipe(
        MXFP8TrainingRecipe.MXFP8_RCEIL
    )

    def eligible_linear(module, _fqn):
        if not isinstance(module, nn.Linear):
            return False
        # Native MXFP8 scales blocks of 32 values. The language-model head has
        # 65 outputs and stays in high precision; transformer matrices qualify.
        return module.in_features % 32 == 0 and module.out_features % 32 == 0

    quantize_(model, mx_config, filter_fn=eligible_linear, device="cuda")
    return model

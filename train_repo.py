from config import Config
from model_repro import Model
from model_repo import apply_mxfp8_training
config = Config()
import numpy as np
import os,random, torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
ddp = int(os.environ.get('RANK', -1)) != -1 # if ddp means torchrun is enabled 

def get_data(split):
    
    if split == 'train':
        data = np.memmap(os.path.join('./data/shakespeare_char', "train.bin"),dtype=np.uint16, mode='r')
        
    else: 
        data = np.memmap(os.path.join('./data/shakespeare_char', "val.bin"),dtype=np.uint16)
    ids = torch.randint(0, len(data) - config.context_length, (config.batch_size,))
    x = torch.stack( [torch.from_numpy(data[idx:idx+config.context_length]
                                                       .astype(np.long)) for idx in ids])
    y = torch.stack( [torch.from_numpy(data[idx+1:idx+config.context_length+1]
                                                       .astype(np.long)) for idx in ids])
    x = x.to(config.device)
    y = y.to(config.device)
    return (x, y)

model = Model(config)
params_names = [param[0] for param in model.named_parameters()]
    
#
#import pdb; pdb.set_trace()
optimizer = torch.optim.AdamW(model.named_parameters(), lr=config.lr, weight_decay=config.weight_decay)
iter = 0
import time
t0=time.time()
if ddp:
    dist.init_process_group(backend='nccl')
    ddp_world_size = int(os.environ.get("WORLD_SIZE", -1))
    ddp_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    ddp_rank = int(os.environ.get("RANK", -1))
    config.device = 'cuda:' + str(ddp_local_rank)
    
    
    torch.cuda.set_device(config.device)
    assert(config.gradient_accumulation_steps % ddp_world_size == 0)
    config.gradient_accumulation_steps //= ddp_world_size
    #model.to(config.device)
    torch.manual_seed(3333+ddp_rank)
    if config.linear_mode == "mxfp8":
        model = apply_mxfp8_training(model)
    if config.compile:
        model = torch.compile(model)
    if ddp_rank == 0: print(params_names)

model.to(config.device)

if ddp:
    model = DDP(model,[ddp_local_rank])
    raw_model = model.module
else: raw_model = model

while iter < config.max_steps:
    iter += 1

    for mini_batch in range(config.gradient_accumulation_steps):
        x, y = get_data('train')
        if mini_batch == config.gradient_accumulation_steps -1:
            model.require_backward_grad_sync = True
        with torch.autocast(device_type=config.device,
                            dtype=torch.bfloat16):
            loss = raw_model(x, y) / config.gradient_accumulation_steps
        #import pdb;pdb.set_trace()
        #loss = loss.to(torch.float32)    
        loss.backward()

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    #import pdb;pdb.set_trace()
    t1 = time.time()
    dt = (t1 - t0)*1000
    t0 = t1
    if  (ddp and ddp_rank == 0): 
        if iter % config.steps_to_eval == 0:
            x, y = get_data('eval')
            loss_val = raw_model(x, y)
            print(f"step: {iter}, train loss: {loss.item():.4f}, time: {dt:.2f}ms, val loss: {loss_val.item():.4f}")
        else: print(f"step: {iter}, train loss: {loss.item():.4f}, time: {dt:.2f}ms")
    
    t0 = t1
    
    
if ddp:
    dist.destroy_process_group()
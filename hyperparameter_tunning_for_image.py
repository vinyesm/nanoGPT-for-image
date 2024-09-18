import importlib
import os
import sys
import time
import math
import pickle
from contextlib import nullcontext
import collections

import numpy as np
from ray import train, tune


from model import GPTConfig, GPT
from data.fashion_kaggle.tokenizer import download_imagenet_256_L, load_imagenet_256_L
from data.fashion_kaggle.tokenizer import decode_from_indices, custom_to_pil



def deep_merge(d1, d2):
    """Recursively merge two dictionaries."""
    for k, v in d2.items():
        if isinstance(v, collections.abc.Mapping) and v:
            d1[k] = deep_merge(d1.get(k, {}), v)
        else:
            d1[k] = v
    return d1

def flatten_dict(d, parent_key='', sep='.'):
    items = {}
    for k, v in d.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

default_config = {
    'io': {
        'out_dir': 'out-fashion-kaggle',  # Unchanged
        'eval_interval': 10,  # Unchanged
        'log_interval': 10,  # Unchanged
        'eval_iters': 200,  # Unchanged
        'eval_only': False,  # Unchanged
        'always_save_checkpoint': True,  # Unchanged
        'init_from': 'scratch',  # Unchanged
    },
    
    'wandb': {
        'wandb_log': True,  # Unchanged
        'wandb_project': 'fashion-kaggle',  # Unchanged
        'wandb_run_name': 'test_hpt_ray',  # Unchanged
        'log_media': True,  # Unchanged
    },

    'data': {
        'data_dtype': np.int64,  # Unchanged
        'dataset': 'fashion_kaggle',  # Unchanged
        'gradient_accumulation_steps': 1,  # Unchanged
        'batch_size': 1,  # Unchanged
        'block_size': 64,  # Unchanged
    },
    
    'model': {
        'meta_vocab_size': 262144,  # Unchanged
        'n_layer': 24,  # Unchanged
        'n_head': 24,  # Unchanged
        'n_embd': 768,  # Unchanged
        'dropout': 0.0,  # Unchanged
        'bias': False,  # Unchanged
    },

    'optimizer': {
        'learning_rate': 6e-4,  # Unchanged
        'weight_decay': 1e-1,  # Unchanged
        'beta1': 0.9,  # Unchanged
        'beta2': 0.95,  # Unchanged
        'grad_clip': 1.0,  # Unchanged
        'max_iters': 600000,  # Added
    },
    
    'lr_decay': {
        'decay_lr': True,  # Unchanged
        'warmup_iters': 2000,  # Unchanged
        'lr_decay_iters': 300,  # Unchanged
        'min_lr': 6e-5,  # Unchanged
    },

    'system': {
        'backend': 'nccl',  # Unchanged
        'device': 'cuda',  # Unchanged
        'dtype': 'bfloat16', # if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16',  # Unchanged
        'compile': True,  # Unchanged
    }
}

def train(config):
    '''
    Train function adapted to use a config dictionary for parameters
    '''
    final_config = deep_merge(default_config.copy(), config)

    import torch
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.distributed import init_process_group, destroy_process_group

    # Extract configurations
    io_config = final_config['io']
    wandb_config = final_config['wandb']
    data_config = final_config['data']
    model_config = final_config['model']
    optimizer_config = final_config['optimizer']
    lr_decay_config = final_config['lr_decay']
    system_config = final_config['system']
    flattened_config = flatten_dict(default_config) # useful for logging
    
    # Extract individual parameters from each config group
    out_dir = io_config['out_dir']
    eval_interval = io_config['eval_interval']
    log_interval = io_config['log_interval']
    eval_iters = io_config['eval_iters']
    eval_only = io_config['eval_only']
    always_save_checkpoint = io_config['always_save_checkpoint']
    init_from = io_config['init_from']

    wandb_log = wandb_config['wandb_log']
    wandb_project = wandb_config['wandb_project']
    wandb_run_name = wandb_config['wandb_run_name']
    log_media = wandb_config['log_media']

    dataset = data_config['dataset']
    data_dtype = data_config['data_dtype']
    gradient_accumulation_steps = data_config['gradient_accumulation_steps']
    batch_size = data_config['batch_size']
    block_size = data_config['block_size']

    meta_vocab_size = model_config['meta_vocab_size']
    n_layer = model_config['n_layer']
    n_head = model_config['n_head']
    n_embd = model_config['n_embd']
    dropout = model_config['dropout']
    bias = model_config['bias']

    learning_rate = optimizer_config['learning_rate']
    weight_decay = optimizer_config['weight_decay']
    beta1 = optimizer_config['beta1']
    beta2 = optimizer_config['beta2']
    grad_clip = optimizer_config['grad_clip']

    decay_lr = lr_decay_config['decay_lr']
    warmup_iters = lr_decay_config['warmup_iters']
    lr_decay_iters = lr_decay_config['lr_decay_iters']
    min_lr = lr_decay_config['min_lr']

    backend = system_config['backend']
    device = system_config['device']
    dtype = system_config['dtype']
    compile_model = system_config['compile']

    # Derived parameters
    ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank  # each process gets a different seed
        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1

    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
    print(f"Tokens per iteration will be: {tokens_per_iter:,}")

    if master_process:
        os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    if log_media:
        download_imagenet_256_L()
        enc = load_imagenet_256_L().to(device)
        os.makedirs("examples", exist_ok=True)
        start_ids = [154737]

        def generate_image(ctx, model, enc, start_ids):
            x = (torch.tensor(start_ids, dtype=torch.int64, device=device)[None, ...])
            with torch.no_grad():
                with ctx:
                    y = model.generate(x, 256 - len(start_ids), temperature=0.8, top_k=200)
            x = y[0].detach().to(dtype=torch.int64, device=device)
            xr = decode_from_indices(enc, x, 1)
            ximg = custom_to_pil(xr[0])
            return ximg

    # Data loading logic remains unchanged
    data_dir = os.path.join('/workspace/nanoGPT-for-image/data', dataset)
    print(f"M: os.getcwd() {os.getcwd()}")
    print(f"M: datadir {data_dir}")
    def get_batch(split):
        if split == 'train':
            data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=data_dtype, mode='r')
        else:
            data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=data_dtype, mode='r')
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        if device_type == 'cuda':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    # Model initialization logic
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                      bias=bias, vocab_size=meta_vocab_size, dropout=dropout)

    if init_from == 'scratch':
        print("Initializing a new model from scratch")
        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)

    model = model.to(device)
    
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    # Initialize optimizer
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

    # Compile model if required
    if compile_model:
        print("Compiling the model... (takes a ~minute)")
        model = torch.compile(model)  # Requires PyTorch 2.0

    # Wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # Logging with wandb
    #TODO properly log config in wandb
    if wandb_log and master_process:
        import wandb
        wandb.init(project=wandb_project, name=wandb_run_name, config=flattened_config)

    # Learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        if it > lr_decay_iters:
            return min_lr
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)


    # Estimating loss
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # Training loop
    X, Y = get_batch('train')
    t0 = time.time()
    iter_num = 0
    best_val_loss = 1e9
    local_iter_num = 0
    raw_model = model.module if ddp else model
    running_mfu = -1.0
    print(f"M: Training loop Device: {torch.cuda.current_device()}")

    while iter_num < optimizer_config['max_iters']:
        # Learning rate adjustment
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Evaluate and log at intervals
        if iter_num % eval_interval == 0 and master_process:
            losses = estimate_loss()
            print(f"Step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            if log_media:
                ximg = generate_image(ctx, model, enc, start_ids)
                ximg.save(f"examples/{iter_num}.jpg")
            if wandb_log:
                log_data = {
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu * 100,
                }
                if log_media:
                    log_data["examples"] = wandb.Image(f"examples/{iter_num}.jpg")
                wandb.log(log_data)

            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                    }
                    print(f"Saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

        if iter_num == 0 and eval_only:
            break

        # Training step (gradient accumulation if required)
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps
            X, Y = get_batch('train')
            scaler.scale(loss).backward()

        # Gradient clipping and optimizer step
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # Logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5:
                mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            print(f"Iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

        iter_num += 1
        local_iter_num += 1

    if ddp:
        destroy_process_group()



search_space = {
    'optimizer': {
        "learning_rate": tune.sample_from(lambda spec: 10 ** (-10 * np.random.rand()))
        },
    'data': {
        "block_size": tune.grid_search([16, 64, 128]),
        },
}

train_with_resources = tune.with_resources(train, {"gpu": 1})

tuner = tune.Tuner(
    train_with_resources,
    param_space=search_space,
    tune_config=tune.TuneConfig(num_samples=1, max_concurrent_trials=1), 
)
results = tuner.fit()
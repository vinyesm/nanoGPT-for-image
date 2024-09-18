# train a miniature image generation model

import numpy as np

# i/o
out_dir = 'out-fashion-kaggle'

# dataset
dataset = 'fashion_kaggle'
data_dtype = np.int64

# model
meta_vocab_size = 262144
n_layer = 24
n_head = 24
n_embd = 768

# wandb logging
wandb_log = True
wandb_project = 'fashion-kaggle'
wandb_run_name = f'gpt_l{n_layer}_h{n_head}_e{n_embd}'
# logging images
log_media = True # if True dumps generated image examples in examples folder (and wandb when activated)

# batch
batch_size = 1
block_size = 64 #2048
gradient_accumulation_steps = 1

# adamw optimizer
max_iters = 300 #600000
lr_decay_iters = 300 #600000

# weight decay params
weight_decay = 1e-1

# eval stuff
eval_interval = 10
eval_iters = 200
log_interval = 10

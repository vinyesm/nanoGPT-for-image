# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-fashion-kaggle-single-image'
# eval_interval = 250 # keep frequent because we'll overfit
# eval_iters = 200
# log_interval = 10 # don't print too too often

# # we expect to overfit on this small dataset, so only save when val improves
# always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'fashion-kaggle'
wandb_run_name = 'debug_on_single_imagenet_256_L'
# wandb_run_name = 'imagenet_256_L'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 1
block_size = 64 #2048
gradient_accumulation_steps = 5 * 8 * 10

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 10
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

# dataset = 'fashion_kaggle'
# gradient_accumulation_steps = 1
# batch_size = 16
# block_size = 256 # context of up to 256 previous characters

# # baby GPT model :)
# n_layer = 6
# n_head = 6
# n_embd = 384
# # n_layer = 12
# # n_head = 12
# # n_embd = 768
# dropout = 0.2

# learning_rate = 1e-3 # with baby networks can afford to go a bit higher
# max_iters = 5000
# lr_decay_iters = 5000 # make equal to max_iters usually
# min_lr = 1e-4 # learning_rate / 10 usually
# beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

# warmup_iters = 100 # not super necessary potentially

# # on macbook also add
# # device = 'cpu'  # run on cpu only
# # compile = False # do not torch compile the model

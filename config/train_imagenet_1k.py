from data.tokenizer import download_imagenet_256_L, load_imagenet_256_L
from data.tokenizer import decode_from_indices, custom_to_pil


init_from='resume' #resume or scratch
# number of parameters: 286.28M

# train a miniature image generation model
meta_vocab_size = 262144
name = 'imagenet1k-all-256-1d-batch16-lr6e-3'
# name = 'imagenet1k-all-256-1d'
out_dir = f'out-{name}'
dataset = 'imagenet_1k'
data_dtype = np.int64

# batch
batch_size = 16
block_size = 255 #64 #2048
gradient_accumulation_steps = 5 * 8 * 10

# max iters
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 50
eval_iters = 200
log_interval = 50

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
positional = "1d"

# wandb
wandb_log = True
wandb_project = 'imagenet1k'
wandb_run_name = name

# weight decay
learning_rate = 6e-3
weight_decay = 1e-1

# to sample images during training and log to wandb
log_media = True # if True dumps generated image examples in examples folder (and wandb when activated)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
download_imagenet_256_L()
enc = load_imagenet_256_L().to(DEVICE)
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
os.makedirs("examples", exist_ok=True)

# for logging generated images during training
start_ids = [0] # a start token (or a list of tokens)
def generate_image(model, enc):
    x = (torch.tensor(start_ids, dtype=torch.int64, device=device)[None, ...])
    with torch.no_grad():
        with ctx:
            y = model.generate(x, 256 - len(start_ids), temperature=0.8, top_k=200)
    x = y[0].detach().to(dtype=torch.int64, device=DEVICE)
    xr = decode_from_indices(enc, x, 1)
    ximg = custom_to_pil(xr[0])
    return ximg

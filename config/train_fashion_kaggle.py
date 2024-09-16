from data.fashion_kaggle.tokenizer import download_imagenet_256_L, load_imagenet_256_L
from data.fashion_kaggle.tokenizer import decode_from_indices, custom_to_pil

# train a miniature image generation model

meta_vocab_size = 262144
out_dir = 'out-fashion-kaggle'
dataset = 'fashion_kaggle'
data_dtype = np.int64

# wandb
wandb_log = True
wandb_project = 'fashion-kaggle'
wandb_run_name = 'babygpt_286M_imagenet_256_L'

# batch
batch_size = 1
block_size = 64 #2048
gradient_accumulation_steps = 5 * 8 * 10

# max iters
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 10
eval_iters = 200
log_interval = 10

# weight decay
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
start_ids = [154737, 116747, 228897,  69697,  93731, 216166,   1176,  54925,  38599,
181844,  84083, 226373,  69154, 224361,  69155, 218209, 223348,  73227,
106855, 228394,  93249, 201056, 171069, 186658, 204514, 104134, 138280,
92197,  94219, 102723,  96294,  91714, 126309,  90146,  24525, 106529,
93794,  95269, 243052, 116385, 121609,  62229,  88097,  87375, 208432,
5161, 121189,  91712, 122145, 223842,  70665, 122661,  97319, 215239,
78160,  87632,  24372, 157296, 215237,  97891, 129349,  75299,  56399,
221793]

def generate_image(model, enc, start_ids):
    x = (torch.tensor(start_ids, dtype=torch.int64, device=device)[None, ...])
    with torch.no_grad():
        with ctx:
            y = model.generate(x, 256 - len(start_ids), temperature=0.8, top_k=200)
    x = y[0].detach().to(dtype=torch.int64, device=DEVICE)
    xr = decode_from_indices(enc, x, 1)
    ximg = custom_to_pil(xr[0])
    return ximg

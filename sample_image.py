"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import PIL
import numpy as np
import torch
from model import GPTConfig, GPT
from data.tokenizer import download_imagenet_256_L, load_imagenet_256_L, decode_from_indices, custom_to_pil

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out-imagenet1k-all-256-1d' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 # number of samples to draw
max_new_tokens = 256 # number of tokens generated in each sample. Each image is 16*16 tokens. 
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda:1' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
sample_dir = f"samples"
exec(open('configurator.py').read()) # overrides from command line or config file
os.makedirs(sample_dir, exist_ok=True)
# -----------------------------------------------------------------------------

# save images and tensors
def to_img(ids, sample_dir, name):
    x = (torch.tensor(ids, dtype=torch.int64, device=device))
    xr = decode_from_indices(enc, x, 1)
    ximg = custom_to_pil(xr[0])
    ximg.save(f"{sample_dir}/{name}_img.jpg")
    # with open(f"{sample_dir}/{name}_ids.pkl", "wb") as pkl:
    #     pickle.dump(ids, pkl)
    # torch.save(x,f"{sample_dir}/{name}.pt")
    # torch.save(xr,f"{sample_dir}/{name}_r.pt")

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda:1' #'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# download and load the image tokenizer model
download_imagenet_256_L()
enc = load_imagenet_256_L().to(device)

m = np.memmap('data/imagenet_1k/test.bin', dtype=np.int64, mode='r')
i = 3 #3,20,50,15000,38000
original = m[i*256:i*256+256]
start_ids = list(original[0:150]) # a start token (or a list of tokens)
# start_ids = [0]

# input
x = (torch.tensor(start_ids, dtype=torch.int64, device=device)[None, ...])
to_img(original, sample_dir, f"original") # save original
to_img(start_ids + [0]* (256 - len(start_ids)), sample_dir, f"input") # save input

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            #generate
            y = model.generate(x, 256 - len(start_ids), temperature=temperature, top_k=top_k)     
            print(y.shape) 
            to_img(y[0], sample_dir, f"{k}_output") # save output
            print('---------------')


# to load
# with open(f"{sample_dir}/xids.pkl", "rb") as pkl:
#     xids = pickle.load(pkl)
#
# x = torch.load(f"{sample_dir}/x.pt", weights_only=True)


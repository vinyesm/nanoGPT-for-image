"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import PIL
import numpy as np
import torch
import tiktoken
from model import GPTConfig, GPT
from data.fashion_kaggle.tokenizer import download_imagenet_256_L, load_imagenet_256_L, decode_from_indices, custom_to_pil

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out-fashion-kaggle' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 # number of samples to draw
max_new_tokens = 256 # number of tokens generated in each sample. Each image is 16*16 tokens. 
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
# if init_from == 'resume':
# init from a model saved in a specific directory
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
# elif init_from.startswith('gpt2'):
#     # init from a given GPT-2 model
#     model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)

#TODO add encode/decode functions from image tokenizer model
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# download the image tokenizer model
download_imagenet_256_L()
enc = load_imagenet_256_L().to(DEVICE)



original = [ 7281, 51205, 24105, 21505, 13923, 51873, 30315,  2561, 24097,
        40961, 52971, 27685,  7713, 23713, 19045, 21601, 26740,  7690,
        41003, 24613, 24099, 29765, 49319, 31843, 29733, 32355, 29733,
        57699, 59907, 13858, 23887, 25185, 60708, 12297, 55237, 15971,
        26690, 52259, 19534, 26804,  5319, 18460, 17638, 23589, 30949,
        57867,  7717, 54597, 52321, 21035, 32292, 22852, 34440, 13192,
        56904, 46674, 36832, 57325, 51738, 54880, 18479, 64613, 23075,
        34377, 22569, 41028, 31779, 16647, 51755, 17249, 48579, 33100,
        37849,  7971, 26480, 31972, 28197, 21549, 43522,  5729, 52525,
        13923, 27146, 21734, 35428, 12124, 50603, 23543, 42615, 24494,
        34759, 56108,  3649, 24879, 31777, 17986, 17453, 41319, 31777,
        20993, 30209,  3936, 40905, 46120, 36545,  3198, 59345,  6213,
        23566,  4971, 59429, 13888,  7200, 54351, 15200, 27655, 20517,
        23908, 33035, 36473, 36714, 24233, 38480,  6181, 31234,  5161,
        55653, 25186, 26656, 17967, 45409, 28194, 29799,   320, 18431,
         2248, 54505, 39771, 20001, 22629, 30534,  9729, 56367, 25184,
        27748,  4618, 56361, 59938, 21637, 36204, 19913, 53057, 34692,
        40427,  1649, 18661, 64807, 18977,  7725, 50497, 52516, 19051,
         5120, 32359, 21604, 11882,  1528,  1856, 36639,  3554, 55241,
        23013, 21517, 59426,  6669, 50533, 31780, 49519, 29225, 27716,
        21861, 38221,  3955, 36811,  1382,  3979, 45902, 24113, 19500,
        57955,  7685, 50283, 26657,  5642, 39265, 31267, 24641, 64737,
        38465, 24769, 48707,  1107, 63729,  7233, 20783, 62823, 27171,
         9792, 51557, 26114, 37899, 55589, 14913,  6756, 23607, 55846,
        31781, 23077, 25892, 51907, 15872, 37931, 55589, 25153, 23721,
        52071, 13825, 52583, 28194, 23593,  6953, 15433, 13153, 32006,
         6152, 15717, 26665, 20069, 11815,  5225, 17444, 62566, 18977,
        30241, 53605, 29218, 58438, 26148, 54379, 25124, 26211, 26148,
        22117, 62498, 27221, 22112]
start_ids = original[0:64]
x = (torch.tensor(start_ids, dtype=torch.int64, device=device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, 256 - len(start_ids), temperature=temperature, top_k=top_k)
            print(y.shape)
            print('---------------')


# save image
os.makedirs("sample", exist_ok=True)
x = (torch.tensor(original, dtype=torch.int64, device=DEVICE))
xr = decode_from_indices(enc, x, 1)
ximg = custom_to_pil(xr[0])
ximg.save("sample/original.jpg")
y = (torch.tensor(y[0], dtype=torch.int64, device=DEVICE))
yr = decode_from_indices(enc, y, 1)
yimg = custom_to_pil(yr[0])
yimg.save("sample/generated.jpg")

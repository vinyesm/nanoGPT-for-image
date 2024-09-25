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
from data.fashion_kaggle.tokenizer import download_imagenet_256_L, load_imagenet_256_L, decode_from_indices, custom_to_pil

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out-fashion-kaggle' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 5 # number of samples to draw
max_new_tokens = 256 # number of tokens generated in each sample. Each image is 16*16 tokens. 
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
sample_dir = f"samples_pe"
exec(open('configurator.py').read()) # overrides from command line or config file
os.makedirs(sample_dir, exist_ok=True)
# -----------------------------------------------------------------------------

# save images and tensors
def to_img(ids, sample_dir, name):
    x = (torch.tensor(ids, dtype=torch.int64, device=DEVICE))
    xr = decode_from_indices(enc, x, 1)
    ximg = custom_to_pil(xr[0])
    ximg.save(f"{sample_dir}/{name}_img.jpg")
    with open(f"{sample_dir}/{name}_ids.pkl", "wb") as pkl:
        pickle.dump(ids, pkl)
    torch.save(x,f"{sample_dir}/{name}.pt")
    torch.save(xr,f"{sample_dir}/{name}_r.pt")

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
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
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
download_imagenet_256_L()
enc = load_imagenet_256_L().to(DEVICE)

original = [136305,  51203, 228897,  70721, 229152, 214119,  66780,  40621,
        103631,  34500, 214359, 228449,  84523, 150625, 216673, 218208,
         92277,  81410, 102763, 228384,  93250,  69957, 232750, 184498,
        195874, 105101, 169061,  96293,  77834,  78081,  96614,  91211,
        118117, 221738,  21517, 106753,  97891,  29796,  85960,  96220,
        220768, 260760,  25622,  97381, 117291, 209953, 122191,  90720,
         78880, 117575,  69665, 122467,  97316, 149703,  16458,  51732,
         95282,  59217, 215492,  93415, 122085,  76289,   7715, 251205,
        224804,  70731,  64289, 214055, 136524, 235124, 121699, 163089,
        215358, 142968, 104194,  49790, 218147, 128071, 207393,  71241,
        117093, 218666,  31809,  22628, 167756, 172177,  44941, 247649,
         47660, 165869, 165264, 188493,  27188,  23597, 256034,  67139,
         78889, 123142, 222823,  83138,  52500, 184952, 231152, 106956,
         34832,  58452, 187240, 217784, 144500,  93445, 121327,  78433,
         51237, 212545,  22537, 103778, 172118, 236908, 143698, 193073,
        182140, 164350, 253030, 230270, 169585, 216149,  30310, 246883,
        103721,  90215,  28199, 151586, 261202, 231929, 118401, 233319,
         73376,  36615, 186493, 211172, 155697, 117861,  97323, 115009,
         70753, 130658, 229060,  65692,  29499, 119384, 164950,  44240,
        116760, 126032,  52092,  97704, 219664, 120971, 209441,   5185,
        116777, 222210, 216679,  82985,  30352,  51020, 172915, 176432,
        136022, 176500, 186205,  81340,  20516,  92229, 126439, 214600,
         23593,  74309, 130147,  66785,   7125, 204412, 100965, 101986,
        110241, 102103, 105049,  72881, 218868,  72779,  55591, 213089,
        246821,  80994,  96869,  84238, 220105, 197688, 167120,  59668,
        126036,  41232, 125648, 228959, 218145,  58375, 208421, 119914,
        100705, 221194, 154663,  16672,  89296, 185909, 172627, 172373,
        168276, 153558, 189228, 221201, 144484,  97381,  88099, 116295,
         56361,  76387,  95237,  94186,   9816, 113764, 135594, 233673,
        219893, 144846,  78577,  94170,  22673,  22797, 241187,  66665,
         82980, 130406, 219751, 222400, 110928, 195205, 132335,  62280,
        167951, 142562, 255492, 259696, 203873, 226406,  96869,  87648]
to_img(original, sample_dir, f"original")

# input
start_ids = original[0:128]
x = (torch.tensor(start_ids, dtype=torch.int64, device=device)[None, ...])
to_img(start_ids + [0]* (256 - len(start_ids)), sample_dir, f"input")

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            #generate
            y = model.generate(x, 256 - len(start_ids), temperature=temperature, top_k=top_k)     
            print(y.shape) 
            to_img(y[0], sample_dir, f"{k}_output")
            print('---------------')


# to load
# with open(f"{sample_dir}/xids.pkl", "rb") as pkl:
#     xids = pickle.load(pkl)
#
# x = torch.load(f"{sample_dir}/x.pt", weights_only=True)


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
sample_dir = f"samples"
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

original = [154737, 116747, 228897,  69697,  93731, 216166,   1176,  54925,  38599,
        181844,  84083, 226373,  69154, 224361,  69155, 218209, 223348,  73227,
        106855, 228394,  93249, 201056, 171069, 186658, 204514, 104134, 138280,
         92197,  94219, 102723,  96294,  91714, 126309,  90146,  24525, 106529,
         93794,  95269, 243052, 116385, 121609,  62229,  88097,  87375, 208432,
          5161, 121189,  91712, 122145, 223842,  70665, 122661,  97319, 215239,
         78160,  87632,  24372, 157296, 215237,  97891, 129349,  75299,  56399,
        221793,  78880, 121163,  78369,  23621, 196707,  86570,  84728,  55190,
        207105,  60090, 132676,  84338,  89317, 254475,   7717, 251237,  92261,
        218154,  59905,  83043,  19821, 169162, 206058,  28028,  94737,  39490,
          5354,  62565,  85095, 129317,  23075, 115785, 121121, 106498,  89699,
         87236, 194147, 148620,  13414,  71108, 168005, 136809,  74976,  29933,
        138260,  95333, 207370,   1603,  70697,  96834, 154661, 132205,  36866,
        146671,  23296, 185965,  99402,   5227, 251629, 199930,  20002,  97733,
        124967,  71233, 116777, 222307,  32517, 248614, 181773, 237400, 197746,
         25678, 128224, 132450, 184605,  50189, 186933,  91173, 122223,  90720,
        103721,  97382,  19047,  82096,  29329, 127704,   4390, 169285, 120418,
         30856, 222525,  81333, 199184, 223301,  64559, 214114,  89129, 122915,
        228931,  65752,  21043,  54100, 149102, 173166, 200937, 206144,  77432,
         81145, 155316,  92165,  87015, 222785,  85024, 108098, 220901,  74849,
         73619,  98172, 235219,  33284,  39513, 233135,  40796,  89260,  91381,
        202835,   7723, 115044,  98345,  80962,  96773,  86282,  31564, 198009,
         63969, 243151, 170687, 244715,  50112,  98014, 214119, 129285,  80417,
         99403, 118117, 221227, 155143,  16864,  87712, 171572, 143490,   2535,
         67203, 138406, 258829, 227233, 149556,  97317, 247851,  83533,  23585,
        109155,  97281,  90412,  14144, 227421, 148899,  76521, 165382, 142796,
         96888,  97338,  20485,  22885, 125987,  67171,  82980, 130150, 219751,
        214208, 258384, 191037, 200667, 254152,  32263, 208291, 192580, 127344,
        199761, 222318, 227941,  87136]
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


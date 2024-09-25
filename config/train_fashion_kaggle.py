from data.fashion_kaggle.tokenizer import download_imagenet_256_L, load_imagenet_256_L
from data.fashion_kaggle.tokenizer import decode_from_indices, custom_to_pil

# train a miniature image generation model

meta_vocab_size = 262144
out_dir = 'out-fashion-kaggle'
dataset = 'fashion_kaggle'
data_dtype = np.int64

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

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
positional = "1d"

# wandb
wandb_log = True
wandb_project = 'fashion-kaggle'
wandb_run_name = f'pe2d_gpt_l{n_layer}_h{n_head}_e{n_embd}'

# weight decay
learning_rate = 6e-5
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
start_ids = [136305,  51203, 228897,  70721, 229152, 214119,  66780,  40621,
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
        103721,  90215,  28199, 151586, 261202, 231929, 118401, 233319]

def generate_image(model, enc):
    x = (torch.tensor(start_ids, dtype=torch.int64, device=device)[None, ...])
    with torch.no_grad():
        with ctx:
            y = model.generate(x, 256 - len(start_ids), temperature=0.8, top_k=200)
    x = y[0].detach().to(dtype=torch.int64, device=DEVICE)
    xr = decode_from_indices(enc, x, 1)
    ximg = custom_to_pil(xr[0])
    return ximg

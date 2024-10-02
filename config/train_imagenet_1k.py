from data.tokenizer import download_imagenet_256_L, load_imagenet_256_L
from data.tokenizer import decode_from_indices, custom_to_pil

# train a miniature image generation model
meta_vocab_size = 262144
name = 'imagenet1k-all-256-1d'
out_dir = f'out-{name}'
dataset = 'imagenet_1k'
data_dtype = np.int64

# batch
batch_size = 1
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
learning_rate = 6e-4
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
# start_ids = [218177,  73085, 152975, 245623, 123396,  74340,  43587, 129073,
#         240416,  38931, 219047,  14085,  79591,  73103,  54087,  88129,
#         217191,  58712,   8433, 247066, 193238, 110992, 176969, 164582,
#         101991, 169686,  33938,  58547,  50742,  42162, 185881, 215127,
#          88167,  81225, 131479, 201651, 151827, 136839, 112032, 160474,
#         170672,  29567, 138197, 217519, 220032, 197575, 128871, 116835,
#         226663,  74841,  86688,  86231, 133038, 246755,  59866,  13527,
#         159645, 251958, 202647, 154784, 250110, 204198,  54787,  88166,
#         118885,  81001, 109466, 234893,  96459,   2067, 136008, 113726,
#         228211,  50282,  41288,  71574,  77960, 144513,  64097, 214343,
#          92230, 118267, 200842,  71555,  64986,  39950, 158930,  62828,
#         246304, 191719,  92011,  13975, 152511, 217563,  54847, 116807,
#          88675,  81004,  86915,  69859, 125770, 164906, 182443, 156105,
#         137981, 100955, 156202,  65163, 205956, 132486,  64033,  82031,
#         218471,  89177, 136872, 255422, 168426,    784, 221712, 159585,
#          66986, 192130,    352, 165213, 104715, 202122, 251479,  88641,
#          88167,  76889,  94630,  58275, 140546,  61618,  15609, 164497,
#         261599, 127387, 110704, 218550,  87172,  33191,  63075, 214119,
#         118819,  89853, 202113,  47243,  93334,  65037,  28144,   6903,
#         149926,  25105, 163117, 160645, 151738, 144533,  64081, 118895,
#          86343,  85118,  29628,  26403,  72899, 174791,  26448, 104975,
#         203662,  37859,  64100, 215998, 230859,  20874, 120353,  84071,
#          89191,  99033,  70562, 161819, 142344, 227340,  23484, 191125,
#          78562, 201168, 111013, 130822, 146053, 214278,  56858,  88133,
#         119143, 224344,  69058,  35589,  66927, 228682, 125940, 153593,
#         105520,  78104, 120378,  72192,  92882, 134891,  64325, 213359,
#          88167,  77085,  86154, 234436, 259017, 210830, 105733, 256644,
#         252167,  89696, 145606,  24512, 128200, 199050, 122387,  22119,
#         222279,  94157,   4502, 104795,  10350, 227026,  18050, 100266,
#         185000,  67090, 102150, 196746, 214922, 204181,  54879,  88135,
#          86119, 253305, 218763, 245138, 134044, 238509, 207489, 147647,
#           2700,  11756, 198877, 101477,   7199, 198603,  55031, 213098]
# start_ids = start_ids[0:150]

start_ids = [0]

def generate_image(model, enc):
    x = (torch.tensor(start_ids, dtype=torch.int64, device=device)[None, ...])
    with torch.no_grad():
        with ctx:
            y = model.generate(x, 256 - len(start_ids), temperature=0.8, top_k=200)
    x = y[0].detach().to(dtype=torch.int64, device=DEVICE)
    xr = decode_from_indices(enc, x, 1)
    ximg = custom_to_pil(xr[0])
    return ximg

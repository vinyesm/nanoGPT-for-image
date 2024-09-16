import os
import numpy as np
import pkg_resources
import subprocess
from einops import rearrange
import PIL
from omegaconf import OmegaConf
from taming.models.lfqgan import VQModel
import torch

ckpt_path = "image_tokenizer"


def download_imagenet_256_L():
    '''
    Downloads the image tokenizer model imagenet_256_L.ckpt if not already downloaded
    '''
    if not os.path.exists(os.path.join(ckpt_path, 'imagenet_256_L.ckpt')):
        os.makedirs(ckpt_path, exist_ok=True)
        url = "https://huggingface.co/TencentARC/Open-MAGVIT2/resolve/main/imagenet_256_L.ckpt"
        output_path = os.path.join(ckpt_path, 'imagenet_256_L.ckpt')
        subprocess.run(["wget", url, "-O", output_path])
    else:
        print("imagenet_256_L.ckpt already exists. Skipping download.")

def load_imagenet_256_L():
    '''
    loads the image tokenizer model
    '''
    config_path = pkg_resources.resource_filename('configs', 'gpu/imagenet_lfqgan_256_L.yaml')
    config = OmegaConf.load(config_path)
    model = VQModel(**config.model.init_args)
    model_path = os.path.join(ckpt_path, 'imagenet_256_L.ckpt')
    print(model_path)
    if ckpt_path is not None:
        sd = torch.load(model_path, map_location="cpu", weights_only=True)["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()

def decode_quant(enc, quant):
    '''
    decode a batch of quantized images
    input: quantized images (N, embed_dim, 16, 16). Tensor values are {+1, -1}
    output: reconstructed images (N, C, H, W)
    '''
    # from embeddings to image
    with torch.no_grad():
        reconstructed_images = enc.decode(quant)
    return reconstructed_images

def decode_from_indices(enc, tokens, batch_size):
    '''
    decode a batch of token ids
    input: token ids (N * 16 * 16)
    output: reconstructed images (N, C, H, W)
    '''
    # from token ids to image
    x = rearrange(tokens, "(b s) -> b s", b=batch_size)
    quant = enc.quantize.get_codebook_entry(x, (batch_size, 16, 16, 18), order='')
    return decode_quant(enc, quant)

def custom_to_pil(x):
    '''
    convert a pytorch tensor to a PIL image
    '''
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.)/2.
    x = x.permute(1,2,0).numpy()
    x = (255*x).astype(np.uint8)
    x = PIL.Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x
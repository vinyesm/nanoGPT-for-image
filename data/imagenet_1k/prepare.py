# saves the imagenet-1k image dataset to a binary file for training, 
# using open-magvit2 image tokenizer
# https://github.com/TencentARC/Open-MAGVIT2/tree/main
# python -m data.imagenet_1k.prepare

import os
import requests
import tarfile

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate as custom_collate
from torch.amp import autocast  # Mixed precision
from tqdm import tqdm
from datasets import load_dataset

from data.tokenizer import load_imagenet_256_L, download_imagenet_256_L


BATCH_SIZE = 64 #128
NUM_WORKERS = 8

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
# num_proc = 1 # 1 gpu
# batch_size = 128

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
# num_proc_load_dataset = 16

# download the image tokenizer model
download_imagenet_256_L()

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_proc_load_dataset = 16
# batch_size = 512


# load the image tokenizer model
enc = load_imagenet_256_L().to("cuda:0") # 1.5G

transform = transforms.Compose([
    transforms.ToTensor(),  # Converts a PIL image or numpy array to a Tensor
    transforms.Lambda(lambda image: (image * 2) - 1)  # Scale pixel values to [-1, +1]
])

if __name__ == '__main__':
    # download and load the dataset
    split_dataset = load_dataset("benjamin-paine/imagenet-1k-256x256", num_proc=num_proc_load_dataset)
    split_dataset['val'] = split_dataset.pop("validation")
    
    # this results in:
    # >>> split_dataset
    # DatasetDict({
    #     train: Dataset({
    #         features: ['image', 'label'],
    #         num_rows: 1281167
    #     })
    #     test: Dataset({
    #         features: ['image', 'label'],
    #         num_rows: 100000
    #     })
    #     val: Dataset({
    #         features: ['image', 'label'],
    #         num_rows: 50000
    #     })
    # })

    # we now want to tokenize the dataset. first define the encoding function
    def process(example):
        image_tensors = [transform(image) for image in example['image']]
        batch = torch.stack(image_tensors).to('cuda:0')
        n = len(batch)
        with torch.no_grad():
            with autocast('cuda'):  # Use mixed precision (not much speed improvement)
                if enc.use_ema:
                    with enc.ema_scope():
                        _, _, ids, _ = enc.encode(batch)
                else:
                    _, _, ids, _ = enc.encode(batch)
        ids_reshaped = ids.view(n, 256)
        lengths =  [256] * n
        out = {'ids': ids_reshaped, 'len': lengths}
        return out

    # tokenize the dataset (this is long, takes ~3h on a RTX4090, best params found batch_size=128 and num_proc=1)
    tokenized = split_dataset.map(
        process,
        remove_columns=['image', 'label'],
        batched=True,
        batch_size=128,
        desc="tokenizing the splits",
        num_proc=1, # only one gpu
    )

    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint64 
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

# train.bin is ~2.5G, validation.bin ~98M, test.bin 176M
# train has ~320M tokens (327,978,752)
# val has ~12M tokens (12,800,000)
# test has ~256M tokens (256,00,000)

# to read the bin files later, e.g. with numpy:
# m = np.memmap('train.bin', dtype=np.int64, mode='r')
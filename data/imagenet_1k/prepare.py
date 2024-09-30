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
from tqdm import tqdm
from datasets import load_dataset

from data.tokenizer import load_imagenet_256_L, download_imagenet_256_L


BATCH_SIZE = 64 #128
NUM_WORKERS = 8

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 1 # 1 gpu
batch_size = 128

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = 16



# download the image tokenizer model
download_imagenet_256_L()

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# load the image tokenizer model
enc = load_imagenet_256_L().to(DEVICE) # 1.5G

transform = transforms.Compose([
    transforms.ToTensor(),  # Converts a PIL image or numpy array to a Tensor
    transforms.Lambda(lambda image: (image * 2) - 1)  # Scale pixel values to [-1, +1]
])

class SingleSampleDataset(Dataset):
    '''
    a dataset of a single sample useful for debugging, i.e. overfitting on one image
    '''
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return 1
    
    def __getitem__(self, index):
        return self.data

if __name__ == '__main__':
    # download and load the dataset
    # split_dataset = load_dataset("benjamin-paine/imagenet-1k-256x256", num_proc=num_proc_load_dataset)
    split_dataset = load_dataset("benjamin-paine/imagenet-1k-256x256", split={'train': 'train[0:1]', 'val': 'validation[0:100]', 'test': 'validation[0:50]'},num_proc=5)
    print(f"Length of train set: {split_dataset['train'].num_rows}")
    print(f"Length of validation set: {split_dataset['val'].num_rows}")
    print(f"Length of test set: {split_dataset['test'].num_rows}")

    # we now want to tokenize the dataset. first define the encoding function
    def process(examples):
        image_tensors = [transform(image) for image in examples['image']]  # Apply transform to each image in the batch
        batch = torch.stack(image_tensors).to(DEVICE)
        n = len(batch)
        with torch.no_grad():
            if enc.use_ema:
                with enc.ema_scope():
                    _, _, ids, _ = enc.encode(batch)
            else:
                _, _, ids, _ = enc.encode(batch)
        ids_reshaped = ids.view(n, 256)
        lengths =  [256] * n
        out = {'ids': ids_reshaped, 'len': lengths}
        del batch, image_tensors, ids
        torch.cuda.empty_cache()  # Optionally clear cache after each batch
        # print(out)
        return out

    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=['label', 'image'],
        batched=True,  # Process in batches
        batch_size=batch_size,  # Set your desired batch size
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint64 
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1 #1024
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

# #
# import numpy as np
# import torch
# from data.tokenizer import load_imagenet_256_L, download_imagenet_256_L, custom_to_pil
# DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# enc = load_imagenet_256_L().to(DEVICE)
# m = np.memmap('data/imagenet_1k/train.bin', dtype=np.uint64, mode='r')
# x = (torch.tensor(m, dtype=torch.int64, device=DEVICE)[None, ...])
# q = enc.quantize.get_codebook_entry(x, (1, 16, 16, 18), order='')

# with torch.no_grad():
#     tensor2 = enc.decode(q)

# reconstructed_image2 = custom_to_pil(tensor2[0])
# reconstructed_image2.save("data/imagenet_1k/train.jpg")
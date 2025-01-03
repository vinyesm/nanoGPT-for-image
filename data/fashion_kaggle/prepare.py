# saves the kaggle fashion image dataset to a binary file for training, 
# using open-magvit2 image tokenizer
# https://github.com/TencentARC/Open-MAGVIT2/tree/main
# python -m data.fashion_kaggle.prepare

import os
import requests
import tarfile

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate as custom_collate
from tqdm import tqdm

from custom import CustomTest, CustomTrain
from data.tokenizer import load_imagenet_256_L, download_imagenet_256_L


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64 #128
NUM_WORKERS = 8


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

def download_and_extract_dataset():
    '''
    downloads and extracts images of  fashion kaggle dataset
    '''
    url = 'https://github.com/rom1504/kaggle-fashion-dalle/releases/download/1.0.0/fashion_kaggle.tar'
    root_dir = "tmp"
    target_dir = os.path.join(root_dir, 'fashion_kaggle')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    dataset_tar = os.path.join(target_dir, 'fashion_kaggle.tar')
    response = requests.get(url, stream=True)
    with open(dataset_tar, 'wb') as f:
        for chunk in tqdm(response.iter_content(chunk_size=1024)):
            if chunk:
                f.write(chunk)

    with tarfile.open(dataset_tar, 'r') as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc="Extracting"):
            tar.extract(member, target_dir)

    extracted_files = sum([len(files) for _, _, files in os.walk(target_dir)])
    print(f"Number of extracted files: {extracted_files}")

def write_img_file_list(in_directory, out_file):
  '''
  list all the *.jpg files in in_directory and write all paths to out_file
  '''
  img_paths = [os.path.abspath(os.path.join(in_directory, f))
    for f in os.listdir(in_directory)
    if f.endswith(".jpg")]
  with open(out_file, 'w') as f:
    for path in sorted(img_paths):
      f.write(path + "\n")

if __name__ == '__main__':
    # download and load the dataset
    if not os.path.exists("tmp/fashion_kaggle/fashion_kaggle"):
        download_and_extract_dataset()
    else:
        "tmp/fashion_kaggle/fashion_kaggle exists, skipping dataset download..."
    write_img_file_list("tmp/fashion_kaggle/fashion_kaggle", "tmp/fashion_kaggle/images_list.txt")
    kaggle_dataset = CustomTest(size=256, test_images_list_file="tmp/fashion_kaggle/images_list.txt")

    # split train/val
    g_cpu = torch.Generator()
    g_cpu.manual_seed(1234)
    train, val = torch.utils.data.random_split(kaggle_dataset, [0.9, 0.1], generator=g_cpu)
    print(f"Length of train set: {len(train)} ({100*len(train)/(len(train)+len(val)):0.0f}%)")
    print(f"Length of validation set: {len(val)} ({100*len(val)/(len(train)+len(val)):0.0f}%)")

    # add a debug dataset with single image
    index = 43853  # You can change this index if needed
    data = kaggle_dataset[index]
    single = SingleSampleDataset(data)
    print(f"Length of single set: {len(single)}")

    # download the image tokenizer model
    download_imagenet_256_L()

    # load the image tokenizer model
    enc = load_imagenet_256_L().to(DEVICE)

    # concatenate all the ids in each dataset into one large file we can use for training
    for split in ["val", "train", "single"]:
        batch_size = BATCH_SIZE
        num_workers = NUM_WORKERS
        dtype = np.int64

        if split == "train":
            dataset = train
            filename = 'data/fashion_kaggle/train.bin'
            total = train.__len__() 
        elif split == "val":
            dataset = val
            filename = 'data/fashion_kaggle/val.bin'
            total = val.__len__()
        elif split == "single":
            dataset = single
            total = single.__len__()
            filename = f'data/fashion_kaggle/single.bin'
            batch_size = 1
            num_workers = 1

        
        arr_len = len(dataset)*16*16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=NUM_WORKERS,
                                collate_fn=custom_collate,
                                shuffle=False,
                                pin_memory=True)

        array_list = []
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(dataloader), total=total//BATCH_SIZE):

                # prepare images to be tokenized
                images = batch["image"].permute(0, 3, 1, 2).to(DEVICE)

                # tokenize the dataset
                if enc.use_ema:
                    with enc.ema_scope():
                        quant, _, tokens, _ = enc.encode(images)
                else:
                    quant, _, tokens, _ = enc.encode(images)

                array_list.append(tokens.cpu().numpy())

        # Write into mmap
        arr[:] = np.concatenate(array_list)
        arr.flush()

    # clean tmp files
    # print("removing tmp files")
    # shutil.rmtree("tmp")

# train.bin is ~79M, val.bin ~8.7M, single.bin 2K
# train has ~10M tokens (10,239,232)
# val has ~1M tokens (1,137,664)
# single has 256 tokens

# to read the bin files later, e.g. with numpy:
# m = np.memmap('train.bin', dtype=np.int64, mode='r')
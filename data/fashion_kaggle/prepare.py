# saves kaggle fashion datset images to binary file for training
import os
import requests
import tarfile
from tqdm import tqdm


# TODO: add necessary utils to load image encoder
# tokenizer https://github.com/TencentARC/Open-MAGVIT2
# chkpt https://huggingface.co/TencentARC/Open-MAGVIT2/blob/main/imagenet_256_B.ckpt
enc = None


if __name__ == '__main__':
    
    def download_and_extract_dataset():
        url = 'https://github.com/rom1504/kaggle-fashion-dalle/releases/download/1.0.0/fashion_kaggle.tar'
        root_dir = os.path.dirname(os.path.abspath(__file__))
        target_dir = os.path.join(root_dir, 'fashiopn_kaggle')
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        dataset_tar = os.path.join(target_dir, 'fashion_kaggle.tar')
        response = requests.get(url, stream=True)
        with open(dataset_tar, 'wb') as f:
            for chunk in tqdm(response.iter_content(chunk_size=1024)):
                if chunk:
                    f.write(chunk)

        with tarfile.open(dataset_tar, 'r') as tar:
            tar.extractall(target_dir)

    # the tar file is 631MB (799M untared) and contains 44441 256x256 images
    download_and_extract_dataset()

    # TODO split train and validation set
    
    # TODO tokenize the dataset

    # TODO save the dataset to a binary file for training
        

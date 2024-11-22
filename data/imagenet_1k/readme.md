## Imagenet 1K resized 256x256

<figure>
    <img src="imagenet.jpg" alt="ImageNet dataset" width="400" />
    <figcaption>ImageNet dataset. Source: <a href="https://cs.stanford.edu/people/karpathy/cnnembed/" target="_blank">Stanford University</a></figcaption>
</figure>

1,431,167 images of (256,256) pixels in total from [imagenet_1k dataset](https://huggingface.co/datasets/evanarlian/imagenet_1k_resized_256)(1,431,167 = 1,281,167 (train) + 50,000 (val) + 100,000 (test)). The image tokenizer used is [Open-MAGVIT2](https://github.com/TencentARC/Open-MAGVIT2/tree/main). Each 256×256 image is tokenized into 16×16(256) tokens. 

After running `prepare.py` (preprocess) we get:

- train.bin is ~2.5G, validation.bin ~98M, test.bin 176M
- train has ~320M tokens (327,978,752)
- val has ~12M tokens (12,800,000)
- train has ~256M tokens (256,00,000)





References:

- [imagenet_1k dataset](https://huggingface.co/datasets/evanarlian/imagenet_1k_resized_256)
- [Language Model Beats Diffusion: Tokenizer is key to visual generation](https://magvit.cs.cmu.edu/v2/)
- [Open-MAGVIT2](https://github.com/TencentARC/Open-MAGVIT2/tree/main), [checkpoint 256×256 ImageNet](https://huggingface.co/TencentARC/Open-MAGVIT2/blob/main/imagenet_256_L.ckpt)


## install

For the tokenizer model you need

```bash
pip install open-magvit2
```
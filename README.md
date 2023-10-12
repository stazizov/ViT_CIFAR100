# Description
A PyTorch implementation of Vision Transformers as described in: `An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale` trained on CIFAR-100 dataset

## Requirements

- torch==2.0.1
- pytorch_lightning==1.9.0
- torchmetrics==0.11.4
- torchvision==0.15.2

## How to run

```bash
pip install -r requirements.txt
python main.py
```

## Monitor metrics 
```bash
tensorboard --logdir logs
```

## Usage

```python
model = LightningVisionTransformer(
    image_size = 384,
    patch_size = 16, 
    in_channels = 3, 
    n_classes = 1000, 
    embedding_dimension=768,
    depth=12,
    n_heads=12,
    mlp_ratio=4.,
    qkv_bias=True,
    proj_p=0.,
    attn_p=0
)
```

## References

- https://arxiv.org/abs/2010.11929
- https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py

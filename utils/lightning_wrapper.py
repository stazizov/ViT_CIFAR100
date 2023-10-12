import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics import Accuracy
from model import VisionTransformer

class LightningVisionTransformer(pl.LightningModule):
    def __init__(
        self,
        image_size=384,
        patch_size=16,
        in_channels=3,
        n_classes=100,
        embedding_dimension=768,
        depth=12,
        n_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        proj_p=0.0,
        attn_p=0.0,
    ):
        super().__init__()
        self.model = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            n_classes=n_classes,
            embedding_dimension=embedding_dimension,
            depth=depth,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_p=proj_p,
            attn_p=attn_p,
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=n_classes)
        # self.accuracy = Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.accuracy(logits, y)
        self.accuracy.reset()
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_accuracy", acc, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
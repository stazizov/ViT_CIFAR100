from utils import LightningVisionTransformer, CIFAR100DataModule
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from config import *

if __name__ == "__main__":
    # Instantiate the LightningVisionTransformer model and the CIFAR100DataModule
    model = LightningVisionTransformer(
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
    data_module = CIFAR100DataModule(batch_size=batch_size, validation_split=validation_split)

    # Create a TensorBoardLogger instance and pass it to the Trainer
    logger = TensorBoardLogger("logs/", name="vision_transformer")
    trainer = pl.Trainer(gpus=1, max_epochs=epochs, logger=logger)

    # Run the training using the trainer object
    trainer.fit(model, data_module)
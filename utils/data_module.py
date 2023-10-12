import torchvision.transforms as transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR100
from torchvision.transforms import (
    RandomApply, RandomOrder, RandomAffine, 
    RandomPerspective, GaussianBlur
)

# DataModule to handle data loading and preprocessing
class CIFAR100DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, validation_split=0.2):
        super(CIFAR100DataModule, self).__init__()
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                RandomOrder(
                    [
                        RandomApply([transforms.RandomRotation(15)], p=0.5),
                        RandomApply([RandomPerspective(distortion_scale=0.2)], p=0.5),
                    ]
                ),
                transforms.ToTensor(),
                transforms.Normalize(0, 1),
            ]
        )

    def prepare_data(self):
        CIFAR100(root="./data", train=True, download=True)
        CIFAR100(root="./data", train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            dataset = CIFAR100(root="./data", train=True, transform=self.transform)
            train_size = int((1 - self.validation_split) * len(dataset))
            val_size = len(dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=4
        )
import torch
import torchio as tio
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, Dataset


class RSNAPre(pl.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32, input_size=(251, 251, 150)):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.input_size = input_size

    def setup(self, stage: Optional[str] = None):

        transform = tio.CropOrPad(self.input_size)
        train_set = tio.datasets.RSNAMICCAI(
            self.data_dir, train=True, transform=transform)
        test_set = tio.datasets.RSNAMICCAI(
            self.data_dir, train=False, transform=transform)

        # Training mode: 90/10% split
        split_ratio = 0.9
        n_train_set = int(len(train_set) * 0.9)
        n_val_set = len(train_set) - n_train_set

        self.train_set, self.val_set = random_split(train_set,
                                                    [n_train_set, n_val_set],
                                                    generator=torch.Generator().manual_seed(1))
        self.test_set = test_set

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)

    def teardown(self, stage: Optional[str] = None):
        # Used to clean-up when the run is finished
        ...

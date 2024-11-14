from pathlib import Path
from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader as TorchDataLoader

from .av2_dataset import Av2Dataset, collate_fn
from exp_methods.forecast_mae_api import fmae_collate_fn


class Av2DataModule(LightningDataModule):
    def __init__(
        self,
        data_root: str,
        data_folder: str,
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        test_batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 8,
        pin_memory: bool = True,
        test: bool = False,
    ) -> None:
        super(Av2DataModule, self).__init__()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.data_folder = data_folder
        self.data_root = Path(data_root)

        self.batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size

        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.test = test

        if data_folder == 'processed':
            self.collate = collate_fn
        elif data_folder == 'forecast-mae':
            self.collate = fmae_collate_fn
        else:
            self.collate = lambda _: exit()
            raise NotImplementedError('no such collate method')

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.test:
            self.train_dataset = Av2Dataset(
                data_root=self.data_root / self.data_folder,
                cached_split="train"
            )
            self.val_dataset = Av2Dataset(
                data_root=self.data_root / self.data_folder,
                cached_split="val"
            )
        else:
            self.test_dataset = Av2Dataset(
                data_root=self.data_root / self.data_folder,
                cached_split="test"
            )

    def train_dataloader(self) -> TorchDataLoader:
        return TorchDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate,
            shuffle=self.shuffle,
        )

    def val_dataloader(self) -> TorchDataLoader:
        return TorchDataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate,
            shuffle=False,
        )

    def test_dataloader(self) -> TorchDataLoader:
        return TorchDataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate,
            shuffle=False,
        )
from pathlib import Path
from typing import Dict, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .av2_processor import Av2Extractor
from .av2_utils import KEYS


class Av2Dataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        cached_split: str = None,
        extractor: Av2Extractor = None,
    ) -> None:
        super(Av2Dataset, self).__init__()

        if cached_split is not None:
            self.data_folder = Path(data_root) / cached_split
            self.file_list = sorted(list(self.data_folder.glob("*.pt")))
            self.load = True
        elif extractor is not None:
            self.extractor = extractor
            self.data_folder = Path(data_root)
            print(f"Extracting data from {self.data_folder}")
            self.file_list = list(self.data_folder.rglob("*.parquet"))
            self.load = False
        else:
            raise ValueError("Either cached_split or extractor must be specified")

        print(f"data root: {data_root}/{cached_split}, total number of files: {len(self.file_list)}")

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, index: Union[int, slice]) -> Dict[str, Union[torch.Tensor, str]]:
        if isinstance(index, slice):
            return [self.file_list[i] for i in range(len(self.file_list))[index]]
        
        return torch.load(self.file_list[index]) \
            if self.load else self.extractor.get_data(self.file_list[index])


def collate_fn(batch) -> Dict[str, torch.Tensor]:
    data = {key: pad_sequence([b[key] for b in batch], batch_first=True) for key in KEYS}

    if "actor_scored" in batch[0]:
        data["actor_scored"] = pad_sequence([b["actor_scored"] for b in batch], batch_first=True)

    for key in ["actor_padding_mask", "lane_padding_mask"]:
        data[key] = pad_sequence([b[key] for b in batch], batch_first=True, padding_value=True)

    data["actor_key_padding_mask"] = data["actor_padding_mask"].all(-1)
    data["lane_key_padding_mask"] = data["lane_padding_mask"].all(-1)
    data["num_actors"] = (~data["actor_key_padding_mask"]).sum(-1)
    data["num_lanes"] = (~data["lane_key_padding_mask"]).sum(-1)

    data["scenario_id"] = [b["scenario_id"] for b in batch]
    data["track_id"] = [b["track_id"] for b in batch]

    data["origin_point"] = torch.cat([b["origin_point"] for b in batch], dim=0)
    data["theta_at_origin"] = torch.cat([b["theta_at_origin"] for b in batch])

    return data
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from .download import fetch

class ISICDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transforms: Callable, cache_dir: Path):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = fetch(row.image_id, self.cache_dir)
        img = Image.open(img_path).convert("RGB")
        img = self.transforms(image=np.array(img))["image"]
        target = torch.tensor(row.target).long()
        return img, target

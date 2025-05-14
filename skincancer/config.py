from pathlib import Path
from dataclasses import dataclass


@dataclass
class Config:
    # paths
    cache_dir: Path = Path("images")
    # data
    csv_path: Path  = Path("folds_13062020.csv")
    n_total: int    = 100
    img_size: int   = 512
    batch_size: int = 16
    num_epochs: int = 10
    num_workers: int = 4
    # model
    model_name: str = "efficientnet_b3a"
    # optim
    lr: float = 1e-4
    # misc
    device: str = "cpu" # "cuda" | "cpu"

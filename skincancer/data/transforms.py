import albumentations as A
from albumentations.pytorch import ToTensorV2

from ..config import Config

cfg = Config()
mean = [0.5]*3
std  = [0.5]*3

def _flip():
    """Compatível com versões antigas do albumentations."""
    try:
        return A.Flip()
    except AttributeError:
        return A.OneOf([A.HorizontalFlip(p=1), A.VerticalFlip(p=1)], p=1)

train_tfms = A.Compose([
    A.LongestMaxSize(cfg.img_size), A.PadIfNeeded(cfg.img_size, cfg.img_size),
    A.RandomRotate90(), _flip(),
    A.RandomBrightnessContrast(0.2, 0.2),
    A.ShiftScaleRotate(0.05, 0.1, 20, border_mode=0),
    A.Normalize(mean, std), ToTensorV2()
])

val_tfms = A.Compose([
    A.LongestMaxSize(cfg.img_size), A.PadIfNeeded(cfg.img_size, cfg.img_size),
    A.Normalize(mean, std), ToTensorV2()
])

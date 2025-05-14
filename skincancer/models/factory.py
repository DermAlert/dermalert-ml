import timm
import torch.nn as nn
from ..config import Config

def create_model(cfg: Config, pretrained: bool = True):
    model = timm.create_model(cfg.model_name, pretrained=pretrained, num_classes=1)
    # BCEWithLogits → saída shape (N,1); já ajustamos
    return model

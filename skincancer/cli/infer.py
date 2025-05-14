# ── src/skincancer/cli/infer.py ────────────────────────────────────────────────
"""
Inference CLI
~~~~~~~~~~~~~
Exemplos de uso

# 1. probabilidade em uma imagem única
python -m skincancer.cli.infer \
       --image path/to/ISIC_0000000.jpg \
       --model model_fold1.pt

# 2. avaliação no fold 3 (usa o split do CSV)
python -m skincancer.cli.infer \
       --csv skincancer/folds_13062020.csv \
       --fold 3 \
       --model model_fold3.pt

# 3. ensemble de todos os folds em uma imagem
python -m skincancer.cli.infer \
       --image path.jpg \
       --model model_fold0.pt model_fold1.pt model_fold2.pt model_fold3.pt model_fold4.pt
"""
from __future__ import annotations
import argparse, os
from pathlib import Path
from typing import Sequence, Optional

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader

from ..config import Config
from ..data.transforms import val_tfms
from ..data.dataset import ISICDataset
from ..models.factory import create_model
from ..engine.evaluate import evaluate


# --------------------------------------------------------------------------- #
# Auxiliar: carrega lista de modelos (pode ser 1 ou vários p/ ensemble)
# --------------------------------------------------------------------------- #
def load_models(model_paths: Sequence[Path], cfg: Config):
    models = []
    for mp in model_paths:
        model = create_model(cfg, pretrained=False)
        model.load_state_dict(torch.load(mp, map_location=cfg.device))
        model.eval().to(cfg.device)
        models.append(model)
    return models


# --------------------------------------------------------------------------- #
# 1. Inference em uma imagem isolada
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_image(img_path: Path, models, cfg: Config) -> float:
    pil = Image.open(img_path).convert("RGB")
    tens = val_tfms(image=np.array(pil))["image"].unsqueeze(0).to(cfg.device)
    probs = []
    for m in models:
        p = torch.sigmoid(m(tens))[0, 0].item()
        probs.append(p)
    return float(np.mean(probs))


# --------------------------------------------------------------------------- #
# 2. Avalia um fold inteiro (AUC) usando DataLoader
# --------------------------------------------------------------------------- #
def evaluate_fold(df: pd.DataFrame, fold: int, models, cfg: Config) -> float:
    vl_df = df[df.fold == fold]
    ds = ISICDataset(vl_df, val_tfms, cfg.cache_dir)
    dl = DataLoader(ds, cfg.batch_size * 2, shuffle=False, num_workers=cfg.num_workers)

    # ensemble precisa de loop manual
    gts, preds = [], []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(cfg.device)
            batch_probs = []
            for m in models:
                batch_probs.append(torch.sigmoid(m(x)).cpu().numpy())
            batch_mean = np.mean(batch_probs, axis=0).ravel()  # média entre modelos
            preds.extend(batch_mean)
            gts.extend(y.numpy())
    from sklearn.metrics import roc_auc_score

    return roc_auc_score(gts, preds)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Inference / evaluation")
    ap.add_argument("--image", type=Path, help="Imagem única para inferência")
    ap.add_argument("--csv", type=Path, help="CSV para avaliação de fold")
    ap.add_argument("--fold", type=int, help="Número do fold de validação")
    ap.add_argument("--model", type=Path, nargs="+", required=True,
                    help="Um ou mais .pt gerados no treino")
    args = ap.parse_args()

    cfg = Config()

    # ── carrega 1+ modelos ──────────────────────────────────────────────────
    models = load_models(args.model, cfg)

    if args.image:
        prob = predict_image(args.image, models, cfg)
        print(f"Probabilidade de melanoma: {prob:.4%}")

    elif args.csv and args.fold is not None:
        df = pd.read_csv(args.csv)
        auc = evaluate_fold(df, args.fold, models, cfg)
        print(f"AUC no fold {args.fold}: {auc:.4%}")

    else:
        ap.error("Use --image OU --csv + --fold")


if __name__ == "__main__":
    main()

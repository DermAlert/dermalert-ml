"""
Exemplo de uso:
    python -m skincancer.cli.train --csv data/folds_13062020.csv
"""
from pathlib import Path
import argparse, os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..config import Config
from ..data.dataset import ISICDataset
from ..data.transforms import train_tfms, val_tfms
from ..models.factory import create_model
from ..engine.train import train_epoch
from ..engine.evaluate import evaluate
from ..utils.metrics import pretty_auc

def train_fold(df, cfg: Config, fold: int):
    tr_df = df[df.fold != fold]
    vl_df = df[df.fold == fold]

    ds_tr = ISICDataset(tr_df, train_tfms, cfg.cache_dir)
    ds_vl = ISICDataset(vl_df, val_tfms, cfg.cache_dir)
    dl_tr = DataLoader(ds_tr, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    dl_vl = DataLoader(ds_vl, cfg.batch_size*2, shuffle=False, num_workers=cfg.num_workers)

    model = create_model(cfg).to(cfg.device)
    pos_w = (tr_df.target==0).sum() / (tr_df.target==1).sum()
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=cfg.device))
    opt  = torch.optim.AdamW(model.parameters(), cfg.lr)

    best_auc, best_state = 0, None
    for epoch in range(cfg.num_epochs):
        loss = train_epoch(model, dl_tr, opt, crit, cfg.device)
        auc  = evaluate(model, dl_vl, cfg.device)
        print(f"[fold {fold}] epoch {epoch+1}/{cfg.num_epochs} – loss {loss:.4f} – AUC {auc:.4f}")
        if auc > best_auc:
            best_auc, best_state = auc, model.state_dict()

    out = Path(f"model_fold{fold}.pt")
    torch.save(best_state, out)
    print(f"Fold {fold} concluído | melhor AUC={best_auc:.4f} | salvo em {out}")
    return best_auc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV com folds")
    ap.add_argument("--only-fold", type=int, default=None, help="Treinar um único fold")
    ap.add_argument("--export", action="store_true", help="Gera TorchScript do fold 0")
    args = ap.parse_args()

    cfg = Config()
    df_full = pd.read_csv(Path(args.csv))

    n_pos = cfg.n_total // 2
    n_neg = cfg.n_total - n_pos

    pos_df = df_full[df_full.target == 1].sample(
        n=min(n_pos, (df_full.target == 1).sum()),
        random_state=42,
    )
    neg_df = df_full[df_full.target == 0].sample(
        n=min(n_neg, (df_full.target == 0).sum()),
        random_state=42,
    )

    df = pd.concat([pos_df, neg_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    df['image_id'] = ["_".join(x.split('_')[:2]) for x in df['image_id']]

    folds = [args.only_fold] if args.only_fold is not None else sorted(df.fold.unique())
    aucs  = [train_fold(df, cfg, f) for f in folds]
    print(pretty_auc(sum(aucs)/len(aucs), aucs))

    if args.export and 0 in folds:
        model = create_model(cfg, pretrained=False)
        model.load_state_dict(torch.load("model_fold0.pt", map_location="cpu"))
        model.eval().cpu()
        torch.jit.script(model).save("skin_risk_classifier.pt")
        print("TorchScript exportado ➜ skin_risk_classifier.pt")

if __name__ == "__main__":
    main()

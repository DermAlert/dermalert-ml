def pretty_auc(mean_auc: float, per_fold: list[float]) -> str:
    return f"AUC médio: {mean_auc:.4f}   |   folds: " + ", ".join(f"{v:.3f}" for v in per_fold)

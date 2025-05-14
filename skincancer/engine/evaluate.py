import torch
from sklearn.metrics import roc_auc_score

def evaluate(model, dataloader, device):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for x, y in dataloader:
            logits = model(x.to(device))
            preds.extend(torch.sigmoid(logits).cpu().numpy().ravel())
            gts.extend(y.numpy())
            print("preds", preds)
            print("gts", gts)
    return roc_auc_score(gts, preds)

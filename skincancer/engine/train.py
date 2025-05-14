import torch
from tqdm.auto import tqdm

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    losses = []
    for x, y in tqdm(dataloader, leave=False):
        x, y = x.to(device), y.float().unsqueeze(1).to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return sum(losses) / len(losses)

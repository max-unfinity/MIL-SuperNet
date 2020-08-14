import torch
from torch import nn
import numpy as np
from tqdm.auto import tqdm


def evaluate(model, val_loader, choice=None, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    criterion = nn.CrossEntropyLoss()
    model.eval()
    val_loss = 0
    val_acc = 0
    tq = tqdm(val_loader, leave=False)
    for x, cls in tq:
        with torch.no_grad():
            if isinstance(model, SuperNet):
                out, _ = model(x.to(device), choice)
            else:
                out = model(x.to(device))
        loss = criterion(out, cls.to(device))
        val_loss += loss.item() / len(val_loader)
        val_acc += accuracy_score(cls, out.cpu().argmax(-1)) / len(val_loader)
        tq.set_postfix(loss=f'{loss.item():.4f}')
    return val_loss, val_acc
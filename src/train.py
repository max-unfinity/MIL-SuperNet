import torch
from torch import nn
import numpy as np
from tqdm.auto import tqdm
import os


def train(model, train_loader, train_subval_loader, epochs, controller=None, gamma_start=0.0, gamma_end=0.0, lr=3e-3, prefix_name=''):

    if isinstance(model, SuperNet):
        gamma_scheduler = GammaScheduler(model, total_steps=epochs*len(train_loader), gamma_start=gamma_start, gamma_end=gamma_end)
    else:
        gamma_scheduler = None
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = []

    for epoch in range(epochs):
        train_loss = 0
        val_loss = 0
        best_val_loss = float('inf')

        # Train on 50k/60k training data
        model.train()
        tq = tqdm(train_loader, leave=False)
        for x, cls in tq:
            if isinstance(model, SuperNet):
                out, choice = model(x.to(device))
            else:
                out = model(x.to(device))
            loss = criterion(out, cls.to(device))
            optimizer.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            train_loss += loss.item() / len(train_loader)
            if gamma_scheduler:
                gamma_scheduler.step()
            if controller:
                controller.update_score(choice, loss.item())
            tq.set_postfix(loss=f'{loss.item():.4f}', norm=f'{norm:.4f}')

        # Evaluate on 10k/60k training data
        model.eval()
        tq = tqdm(train_subval_loader, leave=False)
        for x, cls in tq:
            with torch.no_grad():
                if isinstance(model, SuperNet):
                    out, choice = model(x.to(device))
                else:
                    out = model(x.to(device))
            loss = criterion(out, cls.to(device))
            val_loss += loss.item() / len(train_subval_loader)
            if controller:
                controller.update_score(choice, loss.item())
            tq.set_postfix(loss=f'{loss.item():.4f}')

        history.append([train_loss, val_loss])
        state_dict = {'model': model.state_dict(), 'controller': controller, 'gamma_scheduler': gamma_scheduler, 'history': history}
        torch.save(state_dict, f'{prefix_name}_last.pt')
        if (val_loss < best_val_loss):
            best_val_loss = val_loss
            torch.save(state_dict, f'{prefix_name}_best.pt')

        print(f'{epoch+1:>2} / {epochs:>2}, loss = {train_loss:.4f}, val_loss = {val_loss:.4f}')

    return history
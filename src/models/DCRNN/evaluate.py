import torch
import numpy as np


def evaluate(model, loader, device, mean, std):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred = model(batch_x)                     # (B, horizon, N)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    preds   = np.concatenate(all_preds,   axis=0)    # (total, horizon, N)
    targets = np.concatenate(all_targets, axis=0)

    # Denormalise (first feature / call_count)
    preds_d   = preds   * std + mean
    targets_d = targets * std + mean

    mae  = np.mean(np.abs(preds_d - targets_d))
    rmse = np.sqrt(np.mean((preds_d - targets_d) ** 2))
    ss_res = np.sum((targets_d - preds_d) ** 2)
    ss_tot = np.sum((targets_d - np.mean(targets_d)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    return mae, rmse, r2, preds_d, targets_d

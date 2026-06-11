import torch
import numpy as np

# EVALUATION
def evaluate(model, loader, device, mean, std):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred = model(batch_x)                # (B, N, T_out)
            all_preds.append(pred.squeeze(-1).cpu().numpy())
            all_targets.append(batch_y.squeeze(1).cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    targs = np.concatenate(all_targets, axis=0)

    preds_denorm = preds * std + mean
    targs_denorm = targs * std + mean

    mae = np.mean(np.abs(preds_denorm - targs_denorm))
    rmse = np.sqrt(np.mean((preds_denorm - targs_denorm) ** 2))


    ss_res = np.sum((targs_denorm - preds_denorm) ** 2)
    ss_tot = np.sum((targs_denorm - np.mean(targs_denorm)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    return mae, rmse, r2, preds_denorm, targs_denorm

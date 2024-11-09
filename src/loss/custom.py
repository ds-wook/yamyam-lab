import torch
import torch.nn.functional as F


def svd_loss(pred, true, params, regularization, user_idx, diner_idx, num_users, num_diners):
    true = true.squeeze()
    mse = F.mse_loss(pred, true, reduction='mean')
    penalty = torch.tensor(0., requires_grad=True)
    for param in params:
        if param.shape[0] == num_users:
            param = param[user_idx]
        else:
            param = param[diner_idx]
        penalty = penalty + param.data.norm(dim=1).pow(2).sum() * regularization
    return mse + penalty
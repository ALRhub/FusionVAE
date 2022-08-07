import torch
from config import cfg


def get_noisy_features(ftr, target_in):
    if target_in:
        if ftr.shape[1] > 1:
            return ftr[:, 0:-1]
        return None
    else:
        return ftr

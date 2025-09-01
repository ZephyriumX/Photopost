# photopost/nima/utils.py
import numpy as np

def dist_to_mos(p):
    idx = np.arange(1, 11, dtype=np.float32)
    mean = float(np.sum(p * idx))
    var = float(np.sum(p * (idx - mean) ** 2))
    std = float(np.sqrt(var))
    return mean, std
import torch
import numpy as np
import torch.nn as nn
import os
import random

# Seed everything for reproducible results
def seed_everything(seed):
    """
    Set the seed for various random generators to ensure reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

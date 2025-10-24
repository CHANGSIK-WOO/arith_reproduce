import random
import numpy as np
import torch

def set_random_seed(seed):
    """
    모든 random 관련 라이브러리의 seed를 설정합니다.

    Args:
        seed (int): 설정할 random seed 값
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cudnn 관련 설정 (재현성을 위해)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set to {seed}")
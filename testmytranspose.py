import numpy as np
import pandas as pd
import torch

def mytranspose(x):
    if isinstance(x, np.ndarray):
        if x.ndim == 1:  # 1D vector인 경우
            return x.reshape(-1, 1)
        y = np.empty((x.shape[1], x.shape[0]))
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                y[j, i] = x[i, j]
        return y
    elif isinstance(x, pd.DataFrame):
        return x.transpose()
    elif torch.is_tensor(x):
        return x.t()
    else:
        raise TypeError("지원하지 않는 타입입니다.")

# ------------------ 테스트 ------------------


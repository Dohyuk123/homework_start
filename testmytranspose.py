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

# (3) DataFrame의 경우
D = np.array([1, 2, 3, 4])
E = np.array(["red", "white", "red", np.nan])
F = np.array([True, True, True, False])
mydata3 = pd.DataFrame({"d": D, "e": E, "f": F})
transposed_df = mytranspose(mydata3)
assert isinstance(transposed_df, pd.DataFrame)
assert transposed_df.shape == (3, 4)  # 열이 행으로, 행이 열로 바뀜
print("3번 통과")

# (4) PyTorch Tensor의 경우
np_array = np.array([[1, 2], [3, 4]])
tensor_pt = torch.tensor(np_array)
transposed_tensor = mytranspose(tensor_pt)
assert torch.equal(transposed_tensor, torch.tensor([[1, 3], [2, 4]]))
print("4번 통과")

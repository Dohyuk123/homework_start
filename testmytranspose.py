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

# (1) Matrix의 경우
assert np.array_equal(mytranspose(np.array([[1, 2], [3, 4], [5, 6]])), np.array([[1, 3, 5], [2, 4, 6]]))
assert mytranspose(np.empty((0, 0))).shape == (0, 0)
assert np.array_equal(mytranspose(np.array([[1, 2]])), np.array([[1], [2]]))
assert np.array_equal(mytranspose(np.array([[1], [2]])), np.array([[1, 2]]))
print("1번 통과")

# (2) Vector의 경우
v1 = np.array([1, 2, np.nan, 3])
v1_expected = v1.reshape(-1, 1)
assert np.array_equal(mytranspose(v1), v1_expected, equal_nan=True)

v2 = np.array([np.nan])
v2_expected = v2.reshape(-1, 1)
assert np.array_equal(mytranspose(v2), v2_expected, equal_nan=True)

v3 = np.array([])
v3_expected = v3.reshape(-1, 1)
assert np.array_equal(mytranspose(v3), v3_expected)
print("2번 통과")

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
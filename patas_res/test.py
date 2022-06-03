import numpy as np
import torch


def softmax(x):
    max = np.max(x, axis=1, keepdims=True)  # returns max of each row and keeps same dims
    print("max", max)
    e_x = np.exp(x - max)  # subtracts each row with its max value
    sum = np.sum(e_x, axis=1, keepdims=True)  # returns sum of each row and keeps same dims
    f_x = e_x / sum
    return f_x



queries = np.array([[0.2, 0.4], [.8, .6], [1.0, 1.2]])
keys = np.array([[0.1, 0.7, .9, 1.5], [0.3, 0.5, 1.1, 1.3]])
print(queries)
print(keys)
compatibility = np.dot(queries, keys)
print(compatibility)
mask = np.array([[0, 0, 0, 0], [0, 0, 0, -np.inf], [0, 0, -np.inf, -np.inf]])
masked_compat = compatibility + mask
print(masked_compat)
attention = torch.softmax(torch.tensor(masked_compat), 1)
print(attention)
attention_2 = softmax(masked_compat)
print(attention_2)
# denom = np.sum(np.exp(masked_compat))
# print("denom", denom)
# attention = np.divide(np.exp(masked_compat), denom)
# print(attention)
output = np.dot(attention, keys.transpose())
print(output)




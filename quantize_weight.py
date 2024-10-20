import torch
import numpy as np
import gzip





def save_tensor(tensor, path):
    """保存 tensor 对象到文件"""
    torch.save(tensor, gzip.GzipFile(path, "wb"))


def load_tensor(path):
    """从文件读取 tensor 对象"""
    return torch.load(gzip.GzipFile(path, "rb"))

Q = [15, 14, 13, 14, 14, 14]
path = r'quanti_model.pt'
state_dict = load_tensor("retrain_model3.pt")
# my_dict = {}
# result_num = []
for i, state in enumerate(state_dict):
    print(i)
    print(state)
    # layer_cur = state.split('.')[0][-1]
    # layer_name = layer_cur  # 第几层
    state_dict[state].data = state_dict[state].data * (2 ** Q[i])
save_tensor(state_dict, path)
    # print(weight)
    # for k in range(len(np.asarray(weight.cpu()).reshape(1, -1)[0])):
        # shape_num = np.asarray(weight.cpu()).shape
        # num = np.asarray(weight.cpu()).reshape(1, -1)[0][k] * (2 ** Q[i])





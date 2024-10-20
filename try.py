import gzip
import numpy
from PIL import Image
from PIL import ImageDraw
from net import *


def save_tensor(tensor, path):
    """保存 tensor 对象到文件"""
    torch.save(tensor, gzip.GzipFile(path, "wb"))


def load_tensor(path):
    """从文件读取 tensor 对象"""
    return torch.load(gzip.GzipFile(path, "rb"))


state_dict = load_tensor("retrain_model3.pt")
state_dict2 = load_tensor("retrain_model3before.pt")
state_dict3 = load_tensor("retrain_model3origin.pt")
state_dict4 = load_tensor("retrain_model14.pt")
print(state_dict)
print(state_dict2)
print(state_dict3)
print(state_dict4)

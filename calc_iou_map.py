import os
import gzip
import itertools
from net import *


def load_tensor(path):
    """从文件读取 tensor 对象"""
    return torch.load(gzip.GzipFile(path, "rb"))


def read_batches(base_path):
    for batch in itertools.count():
        path = f"{base_path}.{batch}.pt"
        if not os.path.isfile(path):
            break
        x, (y, mask1, mask2) = load_tensor(path)
        yield x.to(device), (y.to(device), mask1, mask2)


for i in range(34, 114):

    # Exception:  FileNotFoundError
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyModel().to(device)
    calc_accuracy = MyModel.calc_accuracy
    try:
        model.load_state_dict(torch.load(f"retrain_model{i}.pt"))
    except FileNotFoundError:
        try:
            model.load_state_dict(torch.load(f"retrain_modelhighest{i}.pt"))
        except FileNotFoundError:
            continue

    # 检查测试集
    model.eval()
    testing_obj_accuracy_list = []
    testing_cls_accuracy_list = []
    testing_iou_accuracy_list = []
    for batch in read_batches(r".\data\testing_set"):
        batch_x, batch_y = batch
        with torch.no_grad():  # 否则跑不动
            predicted = model(batch_x)
            testing_batch_obj_accuracy, testing_batch_cls_accuracy, testing_batch_iou_accuracy = calc_accuracy(batch_y, predicted)
            testing_obj_accuracy_list.append(testing_batch_obj_accuracy)
            testing_cls_accuracy_list.append(testing_batch_cls_accuracy)
            testing_iou_accuracy_list.append(testing_batch_iou_accuracy)
    testing_obj_accuracy = sum(testing_obj_accuracy_list) / len(testing_obj_accuracy_list)
    testing_cls_accuracy = sum(testing_cls_accuracy_list) / len(testing_cls_accuracy_list)
    testing_iou_accuracy = sum(testing_iou_accuracy_list) / len(testing_iou_accuracy_list)
    print(f"epoch{i}:testing obj accuracy: {testing_obj_accuracy}, cls accuracy: {testing_cls_accuracy}, iou accuracy: {testing_iou_accuracy}")
    torch.cuda.empty_cache()

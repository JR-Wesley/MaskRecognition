import os
import gzip
import itertools
from net import *


def save_tensor(tensor, path):
    """保存 tensor 对象到文件"""
    torch.save(tensor, gzip.GzipFile(path, "wb"))


def load_tensor(path):
    """从文件读取 tensor 对象"""
    return torch.load(gzip.GzipFile(path, "rb"))


# 用于启用 GPU 支持
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MyModel.Anchors = MyModel._generate_anchors()


def train():
    """开始训练"""
    # 创建模型实例
    model = MyModel().to(device)
    # model.load_state_dict(load_tensor("model.pt"), False)  # 可选择读取网络
    model.eval()
    # 创建多任务损失计算器
    loss_function = MyModel.loss_function

    # 创建参数调整器
    optimizer = torch.optim.Adam(model.parameters())  # adam 算法

    # 记录训练集和验证集的正确率变化
    training_obj_accuracy_history = []
    training_cls_accuracy_history = []
    validating_obj_accuracy_history = []
    validating_cls_accuracy_history = []

    # 记录最高的验证集正确率
    validating_obj_accuracy_highest = -1
    validating_cls_accuracy_highest = -1
    validating_accuracy_highest = -1
    validating_accuracy_highest_epoch = 0

    # 读取批次的工具函数
    def read_batches(base_path):
        for batch in itertools.count():
            path = f"{base_path}.{batch}.pt"
            if not os.path.isfile(path):
                break
            x, (y, mask1, mask2) = load_tensor(path)
            yield x.to(device), (y.to(device), mask1, mask2)

    # 计算正确率的工具函数
    calc_accuracy = MyModel.calc_accuracy

    # 开始训练过程
    for epoch in range(1, 10000):
        print(f"epoch: {epoch}")

        # 根据训练集训练并修改参数
        # 切换模型到训练模式，将会启用自动微分，批次正规化 (BatchNorm) 与 Dropout
        model.train()
        training_obj_accuracy_list = []
        training_cls_accuracy_list = []
        for batch_index, batch in enumerate(read_batches(r".\data\training_set")):
            # 划分输入和输出
            batch_x, batch_y = batch
            # 计算预测值
            predicted = model(batch_x)
            # 计算损失
            loss = loss_function(predicted, batch_y)
            # 从损失自动微分求导函数值
            loss.backward()
            # 使用参数调整器调整参数
            optimizer.step()
            # 清空导函数值
            optimizer.zero_grad()
            # 记录这一个批次的正确率，torch.no_grad 代表临时禁用自动微分功能
            with torch.no_grad():
                training_batch_obj_accuracy, training_batch_cls_accuracy = calc_accuracy(batch_y, predicted)
            # 输出批次正确率
            training_obj_accuracy_list.append(training_batch_obj_accuracy)
            training_cls_accuracy_list.append(training_batch_cls_accuracy)
            print(f"epoch: {epoch}, batch: {batch_index}: " +
                f"batch obj accuracy: {training_batch_obj_accuracy}, cls accuracy: {training_batch_cls_accuracy}")
        training_obj_accuracy = sum(training_obj_accuracy_list) / len(training_obj_accuracy_list) # 总
        training_cls_accuracy = sum(training_cls_accuracy_list) / len(training_cls_accuracy_list)
        training_obj_accuracy_history.append(training_obj_accuracy)
        training_cls_accuracy_history.append(training_cls_accuracy)
        print(f"training obj accuracy: {training_obj_accuracy}, cls accuracy: {training_cls_accuracy}")

        # 检查验证集
        # 切换模型到验证模式，将会禁用自动微分，批次正规化 (BatchNorm) 与 Dropout
        model.eval()
        validating_obj_accuracy_list = []
        validating_cls_accuracy_list = []
        for batch in read_batches(r".\data\validating_set"):
            batch_x, batch_y = batch
            predicted = model(batch_x)
            validating_batch_obj_accuracy, validating_batch_cls_accuracy = calc_accuracy(batch_y, predicted)
            validating_obj_accuracy_list.append(validating_batch_obj_accuracy)
            validating_cls_accuracy_list.append(validating_batch_cls_accuracy)
        validating_obj_accuracy = sum(validating_obj_accuracy_list) / len(validating_obj_accuracy_list)
        validating_cls_accuracy = sum(validating_cls_accuracy_list) / len(validating_cls_accuracy_list)
        validating_obj_accuracy_history.append(validating_obj_accuracy)
        validating_cls_accuracy_history.append(validating_cls_accuracy)
        print(f"validating obj accuracy: {validating_obj_accuracy}, cls accuracy: {validating_cls_accuracy}")

        # 记录最高的验证集正确率与当时的模型状态，判断是否在 20 次训练后仍然没有刷新记录
        validating_accuracy = validating_obj_accuracy * validating_cls_accuracy
        if validating_accuracy > validating_accuracy_highest:
            validating_obj_accuracy_highest = validating_obj_accuracy
            validating_cls_accuracy_highest = validating_cls_accuracy
            validating_accuracy_highest = validating_accuracy
            validating_accuracy_highest_epoch = epoch
            torch.save(model.state_dict(), f"retrain_modelhighest{epoch}.pt")
            print("highest validating accuracy updated")
        elif validating_obj_accuracy > 0.9 and validating_cls_accuracy > 0.9:
            # 在 20 次训练后仍然没有刷新记录，结束训练
            torch.save(model.state_dict(), f"retrain_model{epoch}.pt")
            print("epoch saved")

        elif epoch - validating_accuracy_highest_epoch > 20:
            # 在 20 次训练后仍然没有刷新记录，结束训练
            print("stop training because highest validating accuracy not updated in 20 epoches")
            break

    # 使用达到最高正确率时的模型状态
    print(f"highest obj validating accuracy: {validating_obj_accuracy_highest}",
        f"from epoch {validating_accuracy_highest_epoch}")
    print(f"highest cls validating accuracy: {validating_cls_accuracy_highest}",
        f"from epoch {validating_accuracy_highest_epoch}")
    model.load_state_dict(load_tensor("retrain_model.pt"), False)

    # 检查测试集
    testing_obj_accuracy_list = []
    testing_cls_accuracy_list = []
    for batch in read_batches(r".\data\testing_set"):
        batch_x, batch_y = batch
        predicted = model(batch_x)
        testing_batch_obj_accuracy, testing_batch_cls_accuracy = calc_accuracy(batch_y, predicted)
        testing_obj_accuracy_list.append(testing_batch_obj_accuracy)
        testing_cls_accuracy_list.append(testing_batch_cls_accuracy)
    testing_obj_accuracy = sum(testing_obj_accuracy_list) / len(testing_obj_accuracy_list)
    testing_cls_accuracy = sum(testing_cls_accuracy_list) / len(testing_cls_accuracy_list)
    print(f"testing obj accuracy: {testing_obj_accuracy}, cls accuracy: {testing_cls_accuracy}")


if __name__ == "__main__":
    train()

import torch.nn.functional as F
import torch
from torch import nn
from collections import defaultdict
from dynamic_quantize_net import *

# 缩放图片的大小
IMAGE_SIZE = (256, 192)

# 分类列表
# 添加other分类提升标签分类的精确度
CLASSES = ["other", "face", "notface"]
CLASSES_MAPPING = {c: index for index, c in enumerate(CLASSES)}
# 判断是否存在对象使用的区域重叠率的阈值 (另外要求对象中心在区域内)
IOU_POSITIVE_THRESHOLD = 0.30
IOU_NEGATIVE_THRESHOLD = 0.30
# 用于启用 GPU 支持
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def adjust_box_by_offset(candidate_box, offset):
    """根据偏移值调整候选区域"""
    x1, y1, w1, h1 = candidate_box
    x_offset, y_offset, w_offset, h_offset = offset
    w2 = w_offset * w1
    h2 = h_offset * h1
    x2 = x1 + w1 * x_offset - w2 // 2
    y2 = y1 + h1 * y_offset - h2 // 2
    x2 = min(IMAGE_SIZE[0]-1,  x2)
    y2 = min(IMAGE_SIZE[1]-1,  y2)
    w2 = min(IMAGE_SIZE[0]-x2, w2)
    h2 = min(IMAGE_SIZE[1]-y2, h2)
    return x2, y2, w2, h2


def calc_iou(rect1, rect2):
    """计算两个区域重叠部分 / 合并部分的比率 (intersection over union)"""
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    xi = max(x1, x2)
    yi = max(y1, y2)
    wi = min(x1+w1, x2+w2) - xi
    hi = min(y1+h1, y2+h2) - yi
    if wi > 0 and hi > 0: # 有重叠部分
        area_overlap = wi*hi
        area_all = w1*h1 + w2*h2 - area_overlap
        iou = area_overlap / area_all
    else: # 没有重叠部分
        iou = 0
    return iou


class MyModel(nn.Module):
    AnchorSpans = (8,)  # 尺度列表，值为锚点之间的距离
    AnchorScales = (2, 4, 6, 8, 10, 12, 14, 16)
    AnchorAspects = ((1, 1), )  # 锚点对应区域的长宽比例列表
    AnchorOutputs = 1 + 4 + len(CLASSES)  # 每个锚点范围对应的输出数量，是否对象中心 (1) + 区域偏移 (4) + 分类数量
    AnchorTotalOutputs = AnchorOutputs * len(AnchorAspects) * len(AnchorScales)  # 每个锚点对应的输出数量
    ObjScoreThreshold = 0.8  # 认为是对象中心所需要的最小分数
    IOUMergeThreshold = 0.3  # 判断是否应该合并重叠区域的重叠率阈值
    # dynamic_q = [15, 14, 13, 13, 14, 11]

    def __init__(self):
        super(MyModel, self).__init__()
        # 主干网络
        self.conv1 = nn.Conv2d(3,  32, kernel_size=3, stride=2, padding=1, bias=False)  # x/2_x/2
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False)  # x/4_x/4
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False)  # x/8_x/8
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)  # x/8_x/8
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)  # x/8_x/8
        self.conv6 = nn.Conv2d(32, MyModel.AnchorTotalOutputs, kernel_size=3, stride=1, padding=1,bias=False)  # x/8_x/8

    # 前向传播
    def forward(self, x):
        # 主干网络前向传播
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        x5 = F.relu(self.conv5(x4))
        x6 = self.conv6(x5)

        outputs=[]
        outputs.append(x6)

        # 连接所有输出
        # 注意顺序需要与 Anchors 一致,先16*12 再8*6 再4*3
        outputs_flatten = []
        for output in reversed(outputs):
            output = output.permute(0, 2, 3, 1)  # 维度转换
            output = output.reshape(output.shape[0], -1, MyModel.AnchorOutputs)
            outputs_flatten.append(output)
        outputs_all = torch.cat(outputs_flatten, dim=1)

        # 是否对象中心应该在 0 ~ 1 之间，使用 sigmoid 处理
        outputs_all[:, :, :1] = torch.sigmoid(outputs_all[:, :, :1])
        # 分类应该在 0 ~ 1 之间，使用 sigmoid 处理
        outputs_all[:, :, 5:] = torch.sigmoid(outputs_all[:, :, 5:])
        return outputs_all

    def quantization(self, fixed_bits):
        self.quantized_conv1 = QuantizedConv2d(self.conv1, fixed_bit=fixed_bits, decimal_bit=15)  # x/2_x/2
        self.quantized_conv2 = QuantizedConv2d(self.conv2, fixed_bit=fixed_bits, decimal_bit=14)  # x/4_x/4
        self.quantized_conv3 = QuantizedConv2d(self.conv3, fixed_bit=fixed_bits, decimal_bit=13)  # x/8_x/8
        self.quantized_conv4 = QuantizedConv2d(self.conv4, fixed_bit=fixed_bits, decimal_bit=13)  # x/8_x/8
        self.quantized_conv5 = QuantizedConv2d(self.conv5, fixed_bit=fixed_bits, decimal_bit=14)  # x/8_x/8
        self.quantized_conv6 = QuantizedConv2d(self.conv6, fixed_bit=fixed_bits, decimal_bit=11)  # x/8_x/8

    def quantization_forward(self, x):  # ,i):
        # 主干网络前向传播
        x = quantization(x, 16, 15)  # 量化版本的输入

        x1 = self.quantized_conv1.quantization_forward(x)
        x1 = F.relu(x1)

        x2 = self.quantized_conv2.quantization_forward(x1)
        x2 = F.relu(x2)

        x3 = self.quantized_conv3.quantization_forward(x2)
        x3 = F.relu(x3)

        x4 = self.quantized_conv4.quantization_forward(x3)
        x4 = F.relu(x4)

        x5 = self.quantized_conv5.quantization_forward(x4)
        x5 = F.relu(x5)

        x6 = self.quantized_conv6.quantization_forward(x5)
        # x6 = self.quantized_conv6(x5)
        print("input: image shape", x.shape)
        print("conv1:output shape", x1.shape, "; weight:", self.quantized_conv1.conv.weight.shape)
        print("conv2:output shape", x2.shape, "; weight:", self.quantized_conv2.conv.weight.shape)
        print("conv3:output shape", x3.shape, "; weight:", self.quantized_conv3.conv.weight.shape)
        print("conv5:output shape", x5.shape, "; weight:", self.quantized_conv4.conv.weight.shape)
        print("conv6:output shape", x6.shape, "; weight:", self.quantized_conv5.conv.weight.shape)

        # 为了仿真而进行保存
        # x=quantization(x, 16, 8)
        # input_arr = np.asarray(input)
        # np.save("C:\\Users\\Tuwu\\Desktop\\video_input_3\\npy\\"+str(i)+".npy", input_arr)

        # 分支网络前向传播
        outputs = []
        outputs.append(x6)
        # 连接所有输出
        # 注意顺序需要与 Anchors 一致,先16*12 再8*6 再4*3
        outputs_flatten = []
        for output in outputs:
            output = output.permute(0, 2, 3, 1)  # 维度转换
            output = output.reshape(output.shape[0], -1, MyModel.AnchorOutputs)
            outputs_flatten.append(output)
        outputs_all = torch.cat(outputs_flatten, dim=1) # 将list转换为tensor,其他不变
        # 是否对象中心应该在 0 ~ 1 之间，使用 sigmoid 处理
        outputs_all[:, :, :1] = torch.sigmoid(outputs_all[:, :, :1])
        # 分类应该在 0 ~ 1 之间，使用 sigmoid 处理
        outputs_all[:, :, 5:] = torch.sigmoid(outputs_all[:, :, 5:])
        return outputs_all

    # 冻结,所有浮点参数全部转为定点，且计算中所用参数进行固化

    def freeze(self):
        # 主干网络
        self.quantized_conv1.freeze()

        self.quantized_conv2.freeze()

        self.quantized_conv3.freeze()

        self.quantized_conv4.freeze()

        self.quantized_conv5.freeze()

        self.quantized_conv6.freeze()

    @staticmethod
    def _generate_anchors():
        """根据锚点和形状生成锚点范围列表"""
        w, h = IMAGE_SIZE
        anchors = []
        for span in MyModel.AnchorSpans:
            for x in range(0, w, span):
                for y in range(0, h, span):
                    xcenter, ycenter = x + span / 2, y + span / 2
                    for scale in MyModel.AnchorScales:
                        for ratio in MyModel.AnchorAspects:
                            ww = span * scale * ratio[0]
                            hh = span * scale * ratio[1]
                            xx = xcenter - ww / 2
                            yy = ycenter - hh / 2
                            xx = max(int(xx), 0)
                            yy = max(int(yy), 0)
                            ww = min(int(ww), w - xx)
                            hh = min(int(hh), h - yy)
                            anchors.append((xx, yy, ww, hh))
        return anchors

    @staticmethod
    def loss_function(predicted, actual):
        """YOLO 使用的多任务损失计算器"""
        result_tensor, result_isobject_masks, result_nonobject_masks = actual
        objectness_losses = []
        offsets_losses = []
        labels_losses = []
        for x in range(result_tensor.shape[0]): # 对每张图片
            mask_positive = result_isobject_masks[x] # 正样本
            mask_negative = result_nonobject_masks[x] # 负样本
            # 计算是否对象中心的损失，分别针对正负样本计算
            # 因为大部分区域不包含对象中心，这里减少负样本的损失对调整参数的影响
            objectness_loss_positive = nn.functional.mse_loss(
                predicted[x, mask_positive, 0], result_tensor[x, mask_positive, 0])
            objectness_loss_negative = nn.functional.mse_loss(
                predicted[x, mask_negative, 0], result_tensor[x, mask_negative, 0]) * 0.5  # 减少影响
            objectness_losses.append(objectness_loss_positive)
            objectness_losses.append(objectness_loss_negative)
            # 计算区域偏移的损失，只针对正样本计算
            offsets_loss = nn.functional.mse_loss(
                predicted[x, mask_positive, 1:5], result_tensor[x, mask_positive, 1:5])
            offsets_losses.append(offsets_loss)
            # 计算标签分类的损失，分别针对正负样本计算
            labels_loss_positive = nn.functional.binary_cross_entropy(
                predicted[x, mask_positive, 5:], result_tensor[x, mask_positive, 5:])
            labels_loss_negative = nn.functional.binary_cross_entropy(
                predicted[x, mask_negative, 5:], result_tensor[x, mask_negative, 5:]) * 0.5
            labels_losses.append(labels_loss_positive)
            labels_losses.append(labels_loss_negative)
        loss = (
                torch.mean(torch.stack(objectness_losses)) +
                torch.mean(torch.stack(offsets_losses)) +
                torch.mean(torch.stack(labels_losses)))
        return loss

    @staticmethod
    def calc_iou(rect1, rect2):
        """计算两个区域重叠部分 / 合并部分的比率 (intersection over union)"""
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        xi = max(x1, x2)
        yi = max(y1, y2)
        wi = min(x1 + w1, x2 + w2) - xi
        hi = min(y1 + h1, y2 + h2) - yi
        if wi > 0 and hi > 0:  # 有重叠部分
            area_overlap = wi * hi
            area_all = w1 * h1 + w2 * h2 - area_overlap
            iou = area_overlap / area_all
        else:  # 没有重叠部分
            iou = 0
        return iou

    @staticmethod
    def calc_accuracy(actual, predicted):
        """YOLO 使用的正确率计算器，这里只计算是否对象中心与标签分类的正确率，区域偏移不计算"""
        result_tensor, result_isobjects, result_nonobjects = actual

        # 计算是否对象中心的正确率，正样本和负样本的正确率分别计算再平均
        a = result_tensor[:, :, 0]
        p = predicted[:, :, 0] > MyModel.ObjScoreThreshold
        # 有物体 且Confidence大于阈值的预测框个数 / 所有标注框（有物体的anchor）个数
        obj_acc_positive = ((a == 1) & (p == 1)).sum().item() / ((a == 1).sum().item() + 0.00001)
        # 没有物体 且Confidence小于阈值的预测框个数 / 没有物体的anchor个数
        obj_acc_negative = ((a == 0) & (p == 0)).sum().item() / ((a == 0).sum().item() + 0.00001)

        obj_acc = (obj_acc_positive + obj_acc_negative) / 2

        #################################################
        # 计算IOU
        photo_total = 0
        iou_single = 0

        for i in range(result_tensor.shape[0]):
            face = list(sorted(result_isobjects[i]))  # 这张照片中有face的anchor标号
            photo_total += 1  # 一共多少张照片
            iou_total = 0
            iou = 0
            for j in face:
                x, y, w, h = result_tensor[i, j, 1:5]
                ax, ay, aw, ah = predicted[i, j, 1:5]
                iou_total += 1  # 一共多少anchor
                iou += calc_iou((ax, ay, aw, ah), (x, y, w, h))
            iou_single += iou / iou_total  # 单张照片的平均IOU
        iou_acc = iou_single / photo_total

        #################################################

        # 计算标签分类的正确率
        cls_total = 0
        cls_correct = 0
        for x in range(result_tensor.shape[0]):
            mask = list(sorted(result_isobjects[x] + result_nonobjects[x]))
            actual_classes = result_tensor[x, mask, 5:].max(dim=1).indices
            predicted_classes = predicted[x, mask, 5:].max(dim=1).indices
            cls_total += len(mask)
            cls_correct += (actual_classes == predicted_classes).sum().item()
        cls_acc = cls_correct / cls_total
        return obj_acc, cls_acc, iou_acc

    @staticmethod
    def convert_predicted_result(predicted):
        """转换预测结果到 (标签, 区域, 对象中心分数, 标签识别分数) 的列表，重叠区域使用 NMS 算法合并"""
        # 记录重叠的结果区域, 结果是 [ [(标签, 区域, RPN 分数, 标签识别分数)], ... ]
        final_result = []
        i=0
        for anchor, tensor in zip(MyModel.Anchors, predicted):
            i=i+1
            obj_score = tensor[0].item()
            if obj_score <= MyModel.ObjScoreThreshold:
                # 要求对象中心分数超过一定值
                continue
            offset = tensor[1:5].tolist()
            if(offset[0]>1 or offset[0]<0):
                print("x偏移值溢出")
                print(offset[0])
            if(offset[1]>1 or offset[1]<0):
                print("y偏移值溢出")
                print(offset[1])
            offset[0] = max(min(offset[0], 1), 0)  # 中心点 x 的偏移应该在 0 ~ 1 之间
            offset[1] = max(min(offset[1], 1), 0)  # 中心点 y 的偏移应该在 0 ~ 1 之间
            box = adjust_box_by_offset(anchor, offset)
            label_max = tensor[5:].max(dim=0)
            cls_score = label_max.values.item()
            label = label_max.indices.item()
            if label == 0:
                # 跳过非对象分类
                continue
            '''''
            else:
                print(i)
                final_result.append((label, box, obj_score, cls_score))
            '''''
            for index in range(len(final_result)):
                exists_results = final_result[index]
                if any(calc_iou(box, r[1]) > MyModel.IOUMergeThreshold for r in exists_results):
                    exists_results.append((label, box, obj_score, cls_score))
                    break
            else:
                final_result.append([(label, box, obj_score, cls_score)])
        # 合并重叠的结果区域 (使用 对象中心分数 * 标签识别分数 最高的区域为结果区域)
        for index in range(len(final_result)):
            exists_results = final_result[index]
            exists_results.sort(key=lambda r: r[2] * r[3])
            final_result[index] = exists_results[-1]
        return final_result

    @staticmethod
    def fix_predicted_result_from_history(cls_result, history_results):
        """根据历史结果减少预测结果中的误判，适用于视频识别，history_results 应为指定了 maxlen 的 deque"""
        # 要求历史结果中 50% 以上存在类似区域，并且选取历史结果中最多的分类
        history_results.append(cls_result)
        final_result = []
        if len(history_results) < history_results.maxlen:
            # 历史结果不足，不返回任何识别结果
            return final_result
        for label, box, rpn_score, cls_score in cls_result:
            # 查找历史中的近似区域
            similar_results = []
            for history_result in history_results:
                history_result = [(calc_iou(r[1], box), r) for r in history_result]
                history_result.sort(key=lambda r: r[0])
                if history_result and history_result[-1][0] > MyModel.IOUMergeThreshold:
                    similar_results.append(history_result[-1][1])
            # 判断近似区域数量是否过半
            if len(similar_results) < history_results.maxlen // 2:
                continue
            # 选取历史结果中最多的分类
            cls_groups = defaultdict(lambda: [])
            for r in similar_results:
                cls_groups[r[0]].append(r)
            most_common = sorted(cls_groups.values(), key=len)[-1]
            # 添加最多的分类中的最新的结果
            final_result.append(most_common[-1])
        return final_result


if __name__ == "__main__":
    anchors = MyModel._generate_anchors()


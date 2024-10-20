import os
import gzip
import random
import numpy
import pandas
from PIL import Image
import xml.etree.cElementTree as ET
from net import *

# 缩放图片的大小
IMAGE_SIZE = (256, 192)
# 训练使用的数据集路径
DATASET_1_IMAGE_DIR = r".\dataset\archive\images"
DATASET_1_ANNOTATION_DIR = r".\dataset\archive\annotations"
DATASET_2_IMAGE_DIR = r".\dataset\784145_1347673_bundle_archive\train\image_data"
DATASET_2_BOX_CSV_PATH = r".\dataset\784145_1347673_bundle_archive\train\bbox_train.csv"
# 分类列表
# Y添加other分类提升标签分类的精确度
CLASSES = ["other", "face", "notface"]
CLASSES_MAPPING = {c: index for index, c in enumerate(CLASSES)}
# 判断是否存在对象使用的区域重叠率的阈值 (另外要求对象中心在区域内)
IOU_POSITIVE_THRESHOLD = 0.30
IOU_NEGATIVE_THRESHOLD = 0.30

# 用于启用 GPU 支持
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MyModel.Anchors = MyModel._generate_anchors()


def save_tensor(tensor, path):
    """保存 tensor 对象到文件"""
    torch.save(tensor, gzip.GzipFile(path, "wb"))


def load_tensor(path):
    """从文件读取 tensor 对象"""
    return torch.load(gzip.GzipFile(path, "rb"))


def calc_resize_parameters(sw, sh):
    """计算缩放图片的参数"""
    sw_new, sh_new = sw, sh
    dw, dh = IMAGE_SIZE
    pad_w, pad_h = 0, 0
    if sw / sh < dw / dh:
        sw_new = int(dw / dh * sh)
        pad_w = (sw_new - sw) // 2 # 填充左右
    else:
        sh_new = int(dh / dw * sw)
        pad_h = (sh_new - sh) // 2 # 填充上下
    return sw_new, sh_new, pad_w, pad_h


def resize_image(img):
    """缩放图片，比例不一致时填充"""
    sw, sh = img.size
    sw_new, sh_new, pad_w, pad_h = calc_resize_parameters(sw, sh)
    img_new = Image.new("RGB", (sw_new, sh_new))
    img_new.paste(img, (pad_w, pad_h))
    img_new = img_new.resize(IMAGE_SIZE)
    return img_new


def image_to_tensor(img):
    """转换图片对象到 tensor 对象"""
    arr = numpy.asarray(img)
    t = torch.from_numpy(arr)
    t = t.transpose(0, 2)  # 转换维度 H,W,C 到 C,W,H
    t = t / 256.0  # 正规化数值使得范围在 0 ~ 1
    return t


def map_box_to_resized_image(box, sw, sh):
    """把原始区域转换到缩放后的图片对应的区域"""
    x, y, w, h = box
    sw_new, sh_new, pad_w, pad_h = calc_resize_parameters(sw, sh)
    scale = IMAGE_SIZE[0] / sw_new
    x = int((x + pad_w) * scale)
    y = int((y + pad_h) * scale)
    w = int(w * scale)
    h = int(h * scale)
    if x + w > IMAGE_SIZE[0] or y + h > IMAGE_SIZE[1] or w == 0 or h == 0:
        return 0, 0, 0, 0
    return x, y, w, h


def map_box_to_original_image(box, sw, sh):
    """把缩放后图片对应的区域转换到缩放前的原始区域"""
    x, y, w, h = box
    sw_new, sh_new, pad_w, pad_h = calc_resize_parameters(sw, sh)
    scale = IMAGE_SIZE[0] / sw_new
    x = int(x / scale - pad_w)
    y = int(y / scale - pad_h)
    w = int(w / scale)
    h = int(h / scale)
    if x + w > sw or y + h > sh or x < 0 or y < 0 or w == 0 or h == 0:
        return 0, 0, 0, 0
    return x, y, w, h


def calc_iou(rect1, rect2):
    """计算两个区域重叠部分 / 合并部分的比率 (intersection over union)"""
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    xi = max(x1, x2)
    yi = max(y1, y2)
    wi = min(x1+w1, x2+w2) - xi
    hi = min(y1+h1, y2+h2) - yi
    if wi > 0 and hi > 0:  # 有重叠部分
        area_overlap = wi*hi
        area_all = w1*h1 + w2*h2 - area_overlap
        iou = area_overlap / area_all
    else:  # 没有重叠部分
        iou = 0
    return iou


def calc_box_offset(candidate_box, true_box):
    """计算候选区域与实际区域的偏移值，要求实际区域的中心点必须在候选区域中"""
    # 计算实际区域的中心点在候选区域中的位置，范围会在 0 ~ 1 之间
    x1, y1, w1, h1 = candidate_box
    x2, y2, w2, h2 = true_box
    x_offset = ((x2 + w2 // 2) - x1) / w1
    y_offset = ((y2 + h2 // 2) - y1) / h1
    # 计算实际区域长宽相对于候选区域长宽的比例，使用 log 减少过大的值
    w_offset = w2 / w1
    h_offset = h2 / h1
    return (x_offset, y_offset, w_offset, h_offset)


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
    return (x2, y2, w2, h2)


def prepare_save_batch(batch, image_tensors, result_tensors, result_isobjects, result_nonobjects):
    """准备训练 - 保存单个批次的数据"""
    # 按索引值列表生成输入和输出 tensor 对象的函数
    def split_dataset(indices):
        indices_list = indices.tolist()
        image_tensors_splited = torch.stack([image_tensors[x] for x in indices_list])
        result_tensors_splited = torch.stack([result_tensors[x] for x in indices_list])
        result_isobjects_splited = [result_isobjects[x] for x in indices_list]
        result_nonobjects_splited = [result_nonobjects[x] for x in indices_list]
        return image_tensors_splited, (
            result_tensors_splited, result_isobjects_splited, result_nonobjects_splited)

    # 切分训练集 (80%)，验证集 (10%) 和测试集 (10%)
    random_indices = torch.randperm(len(image_tensors))
    training_indices = random_indices[:int(len(random_indices)*0.8)]
    validating_indices = random_indices[int(len(random_indices)*0.8):int(len(random_indices)*0.9):]
    testing_indices = random_indices[int(len(random_indices)*0.9):]
    training_set = split_dataset(training_indices)
    validating_set = split_dataset(validating_indices)
    testing_set = split_dataset(testing_indices)

    # 保存到硬盘
    save_tensor(training_set, f"data/training_set.{batch}.pt")
    save_tensor(validating_set, f"data/validating_set.{batch}.pt")
    save_tensor(testing_set, f"data/testing_set.{batch}.pt")
    print(f"batch {batch} saved")


def prepare():
    """准备训练"""
    # 数据集转换到 tensor 以后会保存在 data 文件夹下
    if not os.path.isdir("data"):
        os.makedirs("data")

    # 加载图片和图片对应的区域与分类列表
    # { (路径, 是否左右翻转): [ 区域与分类, 区域与分类, .. ] }
    # 同一张图片左右翻转可以生成一个新的数据，让数据量翻倍
    box_map = defaultdict(lambda: [])
    for filename in os.listdir(DATASET_1_IMAGE_DIR):
        # 从第一个数据集加载
        xml_path = os.path.join(DATASET_1_ANNOTATION_DIR, filename.split(".")[0] + ".xml")
        if not os.path.isfile(xml_path):
            continue
        tree = ET.ElementTree(file=xml_path)
        objects = tree.findall("object")
        path = os.path.join(DATASET_1_IMAGE_DIR, filename)
        for obj in objects:
            class_name = obj.find("name").text
            x1 = int(obj.find("bndbox/xmin").text)
            x2 = int(obj.find("bndbox/xmax").text)
            y1 = int(obj.find("bndbox/ymin").text)
            y2 = int(obj.find("bndbox/ymax").text)
            if (class_name == "mask_weared_incorrect") or (class_name == "with_mask") or (class_name == "without_mask"):
                # 全部转换为face种类
                class_name = "face"
            box_map[(path, False)].append((x1, y1, x2-x1, y2-y1,  CLASSES_MAPPING[class_name]))
            box_map[(path, True)].append((x1, y1, x2-x1, y2-y1,  CLASSES_MAPPING[class_name]))
    df = pandas.read_csv(DATASET_2_BOX_CSV_PATH)
    for row in df.values:
        # 从第二个数据集加载，这个数据集只包含没有带口罩的图片
        filename, width, height, x1, y1, x2, y2 = row[:7]
        path = os.path.join(DATASET_2_IMAGE_DIR, filename)
        box_map[(path, False)].append((x1, y1, x2-x1, y2-y1, CLASSES_MAPPING["face"]))
        box_map[(path, True)].append((x1, y1, x2-x1, y2-y1, CLASSES_MAPPING["face"]))
    # 打乱数据集 (因为第二个数据集只有不戴口罩的图片)
    box_list = list(box_map.items())
    random.shuffle(box_list)
    print(f"found {len(box_list)} images")

    # 保存图片和图片对应的分类与区域列表
    batch_size = 20
    batch = 0
    image_tensors = []  # 图片列表
    result_tensors = []  # 图片对应的输出结果列表，包含 [ 是否对象中心, 区域偏移, 各个分类的可能性 ]
    result_isobjects = []  # 各个图片的包含对象的区域在 Anchors 中的索引
    result_nonobjects = []  # 各个图片不包含对象的区域在 Anchors 中的索引 (重叠率低于阈值的区域)
    for (image_path, flip), original_boxes_labels in box_list:
        with Image.open(image_path) as img_original: # 加载原始图片
            sw, sh = img_original.size # 原始图片大小
            if flip:
                img = resize_image(img_original.transpose(Image.FLIP_LEFT_RIGHT))
            else:
                img = resize_image(img_original) # 缩放图片
            image_tensors.append(image_to_tensor(img))
        # 生成输出结果的 tensor
        result_tensor = torch.zeros((len(MyModel.Anchors), MyModel.AnchorOutputs), dtype=torch.float)
        result_tensor[:, 5] = 1 # 默认分类为 other
        result_tensors.append(result_tensor)
        # 包含对象的区域在 Anchors 中的索引
        result_isobject = []
        result_isobjects.append(result_isobject)
        # 不包含对象的区域在 Anchors 中的索引
        result_nonobject = []
        result_nonobjects.append(result_nonobject)
        # 根据真实区域定位所属的锚点，然后设置输出结果
        negative_mapping = [1] * len(MyModel.Anchors)
        for box_label in original_boxes_labels:
            x, y, w, h, label = box_label
            if flip:  # 翻转坐标
                x = sw - x - w
            x, y, w, h = map_box_to_resized_image((x, y, w, h), sw, sh) # 缩放实际区域
            if w < 20 or h < 20:
                continue  # 缩放后区域过小
            # 定位所属的锚点
            # 要求:
            # - 中心点落在锚点对应的区域中
            # - 重叠率超过一定值
            x_center = x + w // 2
            y_center = y + h // 2
            matched_anchors = []
            for index, anchor in enumerate(MyModel.Anchors):
                ax, ay, aw, ah = anchor
                is_center = (x_center >= ax and x_center < ax + aw and
                    y_center >= ay and y_center < ay + ah)
                iou = calc_iou(anchor, (x, y, w, h))
                if is_center and iou > IOU_POSITIVE_THRESHOLD:
                    matched_anchors.append((index, anchor))  # 区域包含对象中心并且重叠率超过一定值
                    negative_mapping[index] = 0
                elif iou > IOU_NEGATIVE_THRESHOLD:
                    negative_mapping[index] = 0  # 区域与某个对象重叠率超过一定值，不应该当作负样本
            for matched_index, matched_box in matched_anchors:
                # 计算区域偏移
                offset = calc_box_offset(matched_box, (x, y, w, h))
                # 修改输出结果的 tensor
                result_tensor[matched_index] = torch.tensor((
                    1, # 是否对象中心
                    *offset, # 区域偏移
                    *[int(c == label) for c in range(len(CLASSES))] # 对应分类
                ), dtype=torch.float)
                # 添加索引值
                # 注意如果两个对象同时定位到相同的锚点，那么只有一个对象可以被识别，这里后面的对象会覆盖前面的对象
                if matched_index not in result_isobject:
                    result_isobject.append(matched_index)  # 存入有物体的anchor标号
        # 没有找到可识别的对象时跳过图片
        if not result_isobject:
            image_tensors.pop()
            result_tensors.pop()
            result_isobjects.pop()
            result_nonobjects.pop()
            continue
        # 添加不包含对象的区域在 Anchors 中的索引
        for index, value in enumerate(negative_mapping):
            if value:
                result_nonobject.append(index)  # 存入没有物体的anchor
        # 排序索引列表
        result_isobject.sort()
        # 保存批次
        if len(image_tensors) >= batch_size:
            prepare_save_batch(batch, image_tensors, result_tensors,
                               result_isobjects, result_nonobjects)
            image_tensors.clear()
            result_tensors.clear()
            result_isobjects.clear()
            result_nonobjects.clear()
            batch += 1
    #  不保存剩余的批次
    # if len(image_tensors) > 10:
    #     prepare_save_batch(batch, image_tensors, result_tensors,
    #         result_isobject_masks, result_nonobject_masks)


if __name__ == "__main__":
    prepare()


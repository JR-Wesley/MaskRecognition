import gzip
import numpy
import torch
from PIL import Image
from PIL import ImageDraw
from net import *


def save_tensor(tensor, path):
    """保存 tensor 对象到文件"""
    torch.save(tensor, gzip.GzipFile(path, "wb"))


def load_tensor(path):
    """从文件读取 tensor 对象"""
    return torch.load(gzip.GzipFile(path, "rb"), map_location="cpu")


def calc_resize_parameters(sw, sh):
    """计算缩放图片的参数"""
    sw_new, sh_new = sw, sh
    dw, dh = IMAGE_SIZE
    pad_w, pad_h = 0, 0
    if sw / sh < dw / dh:
        sw_new = int(dw / dh * sh)
        pad_w = (sw_new - sw) // 2  # 填充左右
    else:
        sh_new = int(dh / dw * sw)
        pad_h = (sh_new - sh) // 2  # 填充上下
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
    t = t / 255.0  # 正规化数值使得范围在 0 ~ 1
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


# 缩放图片的大小
IMAGE_SIZE = (256, 192)

# 分类列表
CLASSES = ["other", "face", "notface"]
CLASSES_MAPPING = {c: index for index, c in enumerate(CLASSES)}
# 判断是否存在对象使用的区域重叠率的阈值 (另外要求对象中心在区域内)
IOU_POSITIVE_THRESHOLD = 0.30
IOU_NEGATIVE_THRESHOLD = 0.30

# 用于启用 GPU 支持
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MyModel.Anchors = MyModel._generate_anchors()


def eval_model():
    """使用训练好的模型识别图片"""
    # 创建模型实例，加载训练好的状态，然后切换到验证模式
    model = MyModel().to(device)

    model.load_state_dict(torch.load("retrain_model67.pt", map_location="cpu"))
    model.quantization(16)
    # model.load_state_dict(load_tensor("retrain_model67.pt"), False)

    # 询问图片路径，并显示所有可能是人脸的区域
    while True:
        try:
            image_path = input("please input path:")
            if not image_path:
                continue
            # 构建输入
            with Image.open(image_path) as img_original:  # 加载原始图片
                sw, sh = img_original.size  # 原始图片大小
                img = resize_image(img_original)  # 缩放图片
                img_output = img_original.copy()  # 复制图片，用于后面添加标记
                tensor_in = image_to_tensor(img)
            # 预测输出
            model.freeze()
            predicted = model.quantization_forward(tensor_in.unsqueeze(0).to(device))[0]
            torch.save(model, 'quantized_model.pt')
            final_result = MyModel.convert_predicted_result(predicted)
            # 标记在图片上
            draw = ImageDraw.Draw(img_output)
            for label, box, obj_score, cls_score in final_result:
                x, y, w, h = map_box_to_original_image(box, sw, sh)
                score = obj_score * cls_score
                if score < 0.97:
                    continue
                color = "#FF0000" if CLASSES[label] == "person" else "#00FF00"
                draw.rectangle((x, y, x + w, y + h), outline=color, width=3)
                draw.text((x, y - 10), CLASSES[label], fill=color, stroke_width=6)
                draw.text((x, y + h), f"{score:.2f}", fill=color, stroke_width=6)
                print((x, y, w, h), CLASSES[label], obj_score, cls_score)
            img_output.save("img_output.png")
            print("saved to img_output.png")

        except Exception as e:
            print("error:", e)


if __name__ == "__main__":
    eval_model()

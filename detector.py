import torch
import numpy as np
import cv2
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device


class Detector:
    # 初始化属性
    def __init__(self):
        self.img_size = 640  # 照片尺寸缩放
        self.threshold = 0.4  # 置信度阈值
        self.stride = 1  # 卷积步长
        self.weights = './weights/best.pt'  # 权重模型，采用yolov5官方权重
        # self.weights = './weights/train1.pt'  # 权重模型，采用yolov5自训练权重
        self.device = '0' if torch.cuda.is_available() else 'cpu'  # 训练设备根据配置选择显卡或CPU
        self.device = select_device(self.device)
        model = attempt_load(self.weights, map_location=self.device)  # 导入模型权重，指定训练设备
        model.to(self.device).eval()
        model.float()  # 模型数据类型改为float
        self.m = model  # 存放model方法
        self.names = model.module.names if hasattr(model, 'module') else model.names  # 获得识别标签

    def preprocess(self, img):
        # 复制图像副本，浅复制
        img0 = img.copy()
        # yolov5图像预处理letterbox，等比缩放到img_size大小，不够的地方补充黑边
        img = letterbox(img, new_shape=self.img_size)[0]
        # 对于opencv读取的图像数据来说，储存格式为[B,G,R]，B,G,R为色域比例
        # img[:, :, ::-1]指：[B,G,R]翻转成[R,G,B]
        # 我没特别理解这步的作用，因为貌似不进行颜色信道的翻转也能跑
        # img[:, :, (2, 1, 0)]这样的操作同样可以实现翻转
        # transpose(2, 0, 1)作用是将数据shape从(h,w,3)转变为(3,h,w)，方便计算
        img = img[:, :, ::-1].transpose(2, 0, 1)
        # img进行处理后，作内存连续化
        img = np.ascontiguousarray(img)
        # 将img的numpy数组转变为tensor，两者共享内存，指定在显卡上
        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        # opencv采用的是256级RGB，即每个像素的RGB强度都为0-255的整数，这一步是归一重整
        img /= 255.0
        # 如果img张量的维度是3，则将其扩容，一个RGB图片总是三维的
        # 扩充维度的原因估计是从一张图片的处理，变成对一堆图片的处理
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # 返回原图和预处理后的图
        return img0, img

    def detect(self, im):
        # 获得原图和预处理后的图
        im0, img = self.preprocess(im)
        # 调用方法拿到model
        # 根据github，augment是推理增强
        # model本身是一个函数方法，self.m获得这个方法后进行运算，得到一堆预测坐标
        pred = self.m(img, augment=False)[0]
        pred = pred.float()
        # NMS非极大值抑制，排除大部分预测坐标
        pred = non_max_suppression(pred, self.threshold, 0.4)
        # 空盒子
        boxes = []
        # 对pred中的所有数据
        for det in pred:
            # 如果pred不为空
            if det is not None and len(det):
                # 不同尺寸图片，坐标的映射改写
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()
                # 坐标，置信度，类别序号
                for *x, conf, cls_id in det:
                    lbl = self.names[int(cls_id)]  # 根据序号找到类别
                    if lbl not in ['head']:
                        continue  # 判断是否是这几个类别之一
                    pass
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])
                    boxes.append(
                        (x1, y1, x2, y2, lbl, conf))

        return boxes

    def draw_box(self, image, boxes):
        line_thickness = 2
        list_pts = []
        point_radius = 4  # 碰撞半径

        for (x1, y1, x2, y2, lbl, conf) in boxes:
            color = (0, 255, 0)
            check_point_x = int(x1 + ((x2 - x1) * 0.5))
            check_point_y = int(y1 + ((y2 - y1) * 0.5))
            c1, c2 = (x1, y1), (x2, y2)  # c1是左上角，c2是右下角
            # 用长方形的对角端点画框，框住检测出来的人的位置
            cv2.rectangle(image, c1, c2, color, thickness=line_thickness, lineType=cv2.LINE_AA)
            font_thickness = max(line_thickness - 1, 2)
            t_size = cv2.getTextSize(lbl, 0, 1, 2)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3  # 人框已经画完，现在将c2改为标签框的右上角，此时c1是标签框右下角
            cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # 填满一个长方形色块，作为标签颜色背景
            cv2.putText(image, '{}'.format(lbl), (c1[0], c1[1] - 2), 0, 1,
                        [225, 255, 255], thickness=font_thickness, lineType=cv2.LINE_AA)  # 在色块上写字
            # 定义碰撞带
            list_pts.append([check_point_x - point_radius, check_point_y - point_radius])
            list_pts.append([check_point_x - point_radius, check_point_y + point_radius])
            list_pts.append([check_point_x + point_radius, check_point_y + point_radius])
            list_pts.append([check_point_x + point_radius, check_point_y - point_radius])

            ndarray_pts = np.array(list_pts, np.int32)
            # 人框上的红点
            cv2.fillPoly(image, [ndarray_pts], color=(0, 0, 255))
            # 清空
            list_pts.clear()

        return image

    def closest(self, boxes):
        distance_list = []
        for (x1, y1, x2, y2, label, conf) in boxes:
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            distance = (mid_x - 960)**2 + (mid_y - 540)**2
            distance_list.append(distance)
        value = min(distance_list)
        index = distance_list.index(value)
        return index

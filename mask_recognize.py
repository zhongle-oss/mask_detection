#coding=utf-8
import cv2
import os
import numpy as np
from net.mtcnn import mtcnn
import utils.utils as utils
from net.mobileNet import MobileNet
from keras.applications.imagenet_utils import preprocess_input

class face_rec():
    def __init__(self):
        # 创建mtcnn对象
        # 检测图片中的人脸
        self.mtcnn_model = mtcnn()
        # Mtcnn3个网络的门限
        # Pnet=0.5,Rnet=0.6,Onet=0.8
        self.threshold = [0.5,0.6,0.8]

        # 初始化Mobilenet模型的参数
        self.Crop_HEIGHT = 160   # Mobilenet模型的输入: 将检测到的人脸裁剪到160*160*3
        self.Crop_WIDTH = 160
        self.classes_path = "./model_data/classes.txt"
        self.NUM_CLASSES = 2
        # 加载Mobilent网络模型
        self.mask_model = MobileNet(input_shape=[self.Crop_HEIGHT,self.Crop_WIDTH,3],classes=self.NUM_CLASSES)
        # 加载已经训练好了的mobilent权重
        self.mask_model.load_weights("./logs/last_one.h5")
        self.class_names = self._get_class()  # ["mask","nomask"]

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names


    def recognize(self,draw):
        #-----------------------------------------------#
        #   口罩识别
        #   先定位，再进行人脸区域的口罩识别
        #-----------------------------------------------#
        height,width,_ = np.shape(draw)
        # Opencv读取出来的图片的格式是BGR,要将图片转换成RGB格式用于模型检测
        draw_rgb = cv2.cvtColor(draw,cv2.COLOR_BGR2RGB)

        # 检测人脸
        rectangles = self.mtcnn_model.detectFace(draw_rgb, self.threshold)
        if len(rectangles)==0:
            return

        # 数据类型转换float->int, 因为在用opencv函数画框时要求大小参数为整数
        rectangles = np.array(rectangles,dtype=np.int32)
        # 防止框框的大小超过图片的范围
        rectangles[:,0] = np.clip(rectangles[:,0],0,width)
        rectangles[:,1] = np.clip(rectangles[:,1],0,height)
        rectangles[:,2] = np.clip(rectangles[:,2],0,width)
        rectangles[:,3] = np.clip(rectangles[:,3],0,height)

        # Mtcnn检测到的人脸区域的框框可能是长方形，Mobilent模型的输入要求是160*160的正方形，直接cv2.resize可能会失真
        # 这里以不失真的方式utils.rect2square，将mtcnn检测到的人脸框的长和宽转变成正方形
        # 这里的rectangles_temp用于口罩检测
        rectangles_temp = utils.rect2square(np.array(rectangles,dtype=np.int32))
        # 防止框框的大小超过图片的范围
        rectangles_temp[:,0] = np.clip(rectangles_temp[:,0],0,width)
        rectangles_temp[:,1] = np.clip(rectangles_temp[:,1],0,height)
        rectangles_temp[:,2] = np.clip(rectangles_temp[:,2],0,width)
        rectangles_temp[:,3] = np.clip(rectangles_temp[:,3],0,height)

        # 转化成正方形
        #-----------------------------------------------#
        #   对mtcnn检测到的人脸进行是否佩戴口罩的判断
        #-----------------------------------------------#

        classes_all = []
        for rectangle in rectangles_temp:
            # 获取landmark在小图中的坐标
            landmark = (np.reshape(rectangle[5:15],(5,2)) - np.array([int(rectangle[0]),int(rectangle[1])]))/(rectangle[3]-rectangle[1])*160
            # 利用检测到的人脸框 截取图片中人脸的图像
            crop_img = draw_rgb[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            # 将检测到的人脸图像resize到160*160的大小
            crop_img = cv2.resize(crop_img,(self.Crop_HEIGHT,self.Crop_WIDTH))

            # 利用双眼坐标和图片的中心方式进行人脸对齐
            new_img,_ = utils.Alignment_1(crop_img,landmark)
            # 人脸图像的归一化
            new_img = preprocess_input(np.reshape(np.array(new_img,np.float64),[1,self.Crop_HEIGHT,self.Crop_WIDTH,3]))

            # 使用Mobilnet模型进行是否佩戴口罩的分类预测
            classes = self.class_names[np.argmax(self.mask_model.predict(new_img)[0])]
            classes_all.append(classes)

        rectangles = rectangles[:,0:4]

        #-----------------------------------------------#
        #  在原图中画框~!~
        #-----------------------------------------------#
        for (left, top, right, bottom), c in zip(rectangles,classes_all):
            cv2.rectangle(draw, (left, top), (right, bottom), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(draw, c, (left , bottom - 15), font, 0.75, (255, 255, 255), 2)  
        return draw


if __name__ == "__main__":

    # 创建口罩识别的类
    dududu = face_rec()
    video_capture = cv2.VideoCapture(0)

    while True:

        ret, draw = video_capture.read()
        dududu.recognize(draw)
        cv2.imshow('Video', draw)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

    # 测试图片
    img = "/Users/zhongle/Downloads/mask-recognize-master/test_data/2.jpg"
    image = cv2.imread(img)

    # 将图片传入到口罩识别的类当中
    image=dududu.recognize(image)
    if image is None:
        print('No masks detected!')
    else:
        cv2.imshow("1", image)
    cv2.waitKey(0)
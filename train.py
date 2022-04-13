#coding=utf-8
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.applications.imagenet_utils import preprocess_input
from keras.utils import np_utils,get_file
from keras.optimizers import Adam
from keras import backend as K
from utils.utils import get_random_data
from net.mobileNet import MobileNet
from PIL import Image
import numpy as np
import cv2
K.set_image_dim_ordering('tf')


# 网上下载mobilent的预训练权重，这里我下好了放在model_data里面
# BASE_WEIGHT_PATH = ('https://github.com/fchollet/deep-learning-models/'
#                     'releases/download/v0.6/')

HEIGHT = 160
WIDTH = 160
NUM_CLASSES = 2  # 分2类，一类带口罩，一类不带口罩

# 加灰条
def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (0,0,0))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image


# 对数据集的图片进行数据预处理
# # 图片归一化和其标签处理完之后就可以传入到模型中进行训练了
def generate_arrays_from_file(lines,batch_size,train):
    # 获取总长度，即训练集中总共多少张图片
    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        # 获取一个batch_size大小的数据，也就是8张图片
        for b in range(batch_size):
            if i==0:                          # i=0说明已经经过一个世代循环了，也就是之前取出的8张图片已经训练完了，需要把数据进行一个打乱
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            # 从文件中读取图像
            img = Image.open(r".\data\image\train" + '/' + name)
            if train == True:
                # 图像数据增强get_random_data，让网络变得更加有鲁棒性
                img = np.array(get_random_data(img,[HEIGHT,WIDTH]),dtype = np.float64)   # 生成训练数据
            else:
                # 不失真的情况下改变（resize）我们输入图片的大小（长和宽）为模型输入的要求
                img = np.array(letterbox_image(img,[HEIGHT,WIDTH]),dtype = np.float64)  # 生成验证数据

            # 将这张增强后的图片数据保存到X_train里面
            X_train.append(img)
            # 对这张图片的标签进行处理
            Y_train.append(lines[i].split(';')[1])
            # 读完一个周期后重新开始
            i = (i+1) % n

        # 处理图像，图像的归一化
        X_train = preprocess_input(np.array(X_train).reshape(-1,HEIGHT,WIDTH,3))

        # 对图片的标签进行处理，转换成one_hot的形式
        Y_train = np_utils.to_categorical(np.array(Y_train),num_classes= NUM_CLASSES)   

        yield (X_train, Y_train)  # 图片归一化和其标签处理完之后就可以传入到模型中进行训练了


if __name__ == "__main__":
    # 设置模型保存的位置
    log_dir = "./logs/"

    # 打开数据集的txt
    with open(r".\data\train.txt","r") as f:
        lines = f.readlines()

    # 打乱行shuffle，这个txt主要用于帮助读取数据来训练
    # 打乱的数据更有利于训练
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    # 划分验证集和数据集
    # 90%用于训练，10%用于估计。
    num_val = int(len(lines)*0.1)
    num_train = len(lines) - num_val

    # 数据的预处理
    # 图片的归一化处理，标签的预处理，在后面的generate_arrays_from_file函数

    # 建立MobileNet模型 输入：160*160*3的图片
    model = MobileNet(input_shape=[HEIGHT,WIDTH,3],classes=NUM_CLASSES)

    # 去网上下载Mobilent的预训练权重，这里使用我下载好了的权重
    # model_name = 'mobilenet_1_0_224_tf_no_top.h5'
    # weight_path = BASE_WEIGHT_PATH + model_name
    # weights_path = get_file(model_name, weight_path, cache_subdir='models')
    weights_path = "D:\Project\mask-recognize-master\model_data\mobilenet_1_0_224_tf_no_top.h5"
    model.load_weights(weights_path,by_name=True)

    # 保存模型权重的方式，3世代保存一次
    checkpoint_period1 = ModelCheckpoint(
                                    log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                    monitor='acc', 
                                    save_weights_only=False, 
                                    save_best_only=True, 
                                    period=3
                                )

    # 学习率下降的方式，acc精确度三次不下降就下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(
                            monitor='acc', 
                            factor=0.5, 
                            patience=3, 
                            verbose=1
                        )

    # 是否需要早停，当连续10个世代val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = EarlyStopping(
                            monitor='val_loss', 
                            min_delta=0, 
                            patience=10, 
                            verbose=1
                        )

    # 定义训练的loss函数和优化器
    # 交叉熵，这里学习率较高，是比较粗略的训练
    model.compile(loss = 'categorical_crossentropy',
            optimizer = Adam(lr=1e-3),
            metrics = ['accuracy'])

    # 一次的训练集大小
    # 一次训练传入8张图片,可根据自己电脑的配置来调
    batch_size = 16

    # 正式开始训练
    model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size, True),  # 训练trains数据
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=generate_arrays_from_file(lines[num_train:], batch_size, False),  # 训练val数据
            validation_steps=max(1, num_val // batch_size),
            epochs=10,
            initial_epoch=0,
            callbacks=[checkpoint_period1, reduce_lr])

    model.save_weights(log_dir+'middle_one.h5')

    # 交叉熵，梯度下降反向传播
    # 这里的学习率比较低1e-4是比较的精确的训练,但是这里进行粗略的训练就有一个比较好的检测效果
    # 这里训练的比较快，因为网络输入比较小时160*160，训练3个世代，验证集的loss非常小，准确率acc比较高，这是归功于数据增强的功能，数据增强函数让整个网络变得更加具有鲁棒性了
    model.compile(loss = 'categorical_crossentropy',
            optimizer = Adam(lr=1e-4),
            metrics = ['accuracy'])

    # 开始训练
    model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size, True),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=generate_arrays_from_file(lines[num_train:], batch_size, False),
            validation_steps=max(1, num_val//batch_size),
            epochs=20,
            initial_epoch=10,
            callbacks=[checkpoint_period1, reduce_lr])

    model.save_weights(log_dir+'last_one.h5')




import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras


from Datatset import label_image
import config
from model import VisionTransformer
from util import LearningRateSchedule


def testModel(model):
    model.training = False

    # perpare data
    df_label = pd.read_csv(config.TEST_PATH)
    file_paths = df_label['file_name'].values
    labels = df_label['label'].values

    # testing
    total_num = int(len(labels) / 2)
    corrent_num = 0
    for i in range(total_num):
        img_file = file_paths[i]
        label = int(labels[i])

        # 对图片的处理
        img = tf.io.read_file(img_file)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (config.CROP, config.CROP))
        img = tf.cast(img, dtype=config.DTYPE) / 255.0
        img = tf.expand_dims(img, axis=0)
        mean = tf.expand_dims(tf.convert_to_tensor([0.4914, 0.4822, 0.4465], dtype=config.DTYPE), axis=0)
        std = tf.expand_dims(tf.convert_to_tensor([0.2471, 0.2435, 0.2616], dtype=config.DTYPE), axis=0)
        img = (img - mean) / std

        # 网络
        output = model(img, training=False)
        output = tf.nn.softmax(output)
        class_index = tf.squeeze(tf.math.argmax(output, axis=1))

        if class_index == label:
            corrent_num += 1
    accuracy = float(corrent_num) / float(total_num) * 100.
    model.training = True
    return accuracy


if __name__ == '__main__':
    # build model
    vision_transformer = VisionTransformer(
        num_classes=config.NUM_CLASSES,
        num_layers=1,
        train=False,
        resnet=None,
        patches=config.SIZE,
        hidden_size=config.HIDDEN_SIZE,
        representation_size=None,
        classifier='token').model()

    vision_transformer.load_weights(config.LOAD_PATH)


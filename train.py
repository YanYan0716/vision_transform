import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import tensorflow as tf


from Datatset import label_image
import config
from model import VisionTransformer


if __name__ == '__main__':
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # train dataset
    df_train = pd.read_csv(config.TRAIN_PATH)
    file_paths = df_train['name'].values
    labels = df_train['label'].values
    ds_train = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds_train = ds_train \
        .map(label_image, num_parallel_calls=AUTOTUNE) \
        .batch(config.BATCH_SIZE) \
        .shuffle(config.SHUFFLE_BUFFER)\
        .prefetch(AUTOTUNE)

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

    # loss function
    # optimizer
    # training ...
    for i in range(config.TOTAL_STEPS):
        for batch_idx,
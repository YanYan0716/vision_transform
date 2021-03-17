import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras

from Datatset import label_image
import config
from model import VisionTransformer
from util import LearningRateSchedule
from test import testModel

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
        .shuffle(config.SHUFFLE_BUFFER) \
        .prefetch(AUTOTUNE)

    # build model
    vision_transformer = VisionTransformer(
        num_classes=config.NUM_CLASSES,
        num_layers=config.NUM_LAYERS,
        train=False,
        resnet=None,
        patches=config.SIZE,
        hidden_size=config.HIDDEN_SIZE,
        representation_size=None,
        classifier='token').model()

    # loss function
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # lr schedule
    lr_schedule = LearningRateSchedule(
        total_step=config.TOTAL_STEPS,
        base=config.BASE_LR,
        decay_type=config.DECAY_TYPE,
        warmup_steps=config.WARMUP_STEPS
    )

    # training ...
    print('training ')
    step = 0
    BAcc = 0
    for i in range(config.TOTAL_EPOCHS):
        AvgLoss = 0
        for batch_idx, (img, label) in enumerate(ds_train):
            step += 1
            lr = lr_schedule.__call__(step)
            optimizer = keras.optimizers.Adam(learning_rate=lr, beta_1=0.9)
            with tf.GradientTape() as tape:
                output = vision_transformer(img, training=True)
                loss = loss_fn(label, output)
                AvgLoss += loss
            gradients = tape.gradient(loss, vision_transformer.trainable_weights)
            optimizer.apply_gradients(zip(gradients, vision_transformer.trainable_weights))
            # show loss
            if batch_idx % config.LOG_LOSS == 0:
                AvgLoss = AvgLoss / float(config.LOG_LOSS)
                print(f'[epoch: %4d/ ' % i + 'EPOCHS: %4d]\t' % config.TOTAL_EPOCHS +
                      '[step: %6d/ ' % step + 'STEPS: %6d]\t' % config.TOTAL_STEPS +
                      '[loss:%.4f' % AvgLoss)
                AvgLoss = 0

        if i % config.LOG_EPOCH == 0:
            acc = testModel(vision_transformer)
            if acc > BAcc:
                vision_transformer.save_weights(config.SAVE_PATH)
                BAcc = acc
                print(f'saved path: {config.SAVE_PATH}')
            print(f'[epoch: %4d/ ' % i + 'EPOCHS:%4d]\t' % config.TOTAL_EPOCHS +
                  + '[acc:%.4f' % acc + ', BAcc:%.4f]' % BAcc)

# import os
#
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# import pandas as pd
# import tensorflow as tf
# import tensorflow.keras as keras
#
# from Datatset import label_image
# import config
# from model import VisionTransformer
# from util import LearningRateSchedule
# from test import testModel
#
# if __name__ == '__main__':
#     AUTOTUNE = tf.data.experimental.AUTOTUNE
#
#     # train dataset
#     df_train = pd.read_csv(config.TRAIN_PATH)
#     file_paths = df_train['name'].values
#     labels = df_train['label'].values
#     ds_train = tf.data.Dataset.from_tensor_slices((file_paths, labels))
#     ds_train = ds_train \
#         .map(label_image, num_parallel_calls=AUTOTUNE) \
#         .batch(config.BATCH_SIZE) \
#         .shuffle(config.SHUFFLE_BUFFER) \
#         .prefetch(AUTOTUNE)
#
#     # build model
#     vision_transformer = VisionTransformer(
#         num_classes=config.NUM_CLASSES,
#         num_layers=config.NUM_LAYERS,
#         train=True,
#         resnet=None,
#         patches=config.SIZE,
#         hidden_size=config.HIDDEN_SIZE,
#         representation_size=None,
#         classifier='token').model()
#
#     # loss function
#     loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#
#     # lr schedule
#     lr_schedule = LearningRateSchedule(
#         total_step=config.TOTAL_STEPS,
#         base=config.BASE_LR,
#         decay_type=config.DECAY_TYPE,
#         warmup_steps=config.WARMUP_STEPS
#     )
#
#     # training ...
#     print('training ... ')
#     step = 0
#     BAcc = 0
#     for i in range(config.TOTAL_EPOCHS):
#         AvgLoss = 0
#         for batch_idx, (img, label) in enumerate(ds_train):
#             step += 1
#             lr = lr_schedule.__call__(step)
#             optimizer = keras.optimizers.Adam(learning_rate=lr, beta_1=0.9)
#             with tf.GradientTape() as tape:
#                 output = vision_transformer(img, training=True)
#                 loss = loss_fn(label, output)
#                 AvgLoss += loss
#             gradients = tape.gradient(loss, vision_transformer.trainable_weights)
#             optimizer.apply_gradients(zip(gradients, vision_transformer.trainable_weights))
#             # show loss
#             if batch_idx % config.LOG_LOSS == 0:
#                 AvgLoss = AvgLoss / float(config.LOG_LOSS)
#                 print(f'[epoch: %4d/ ' % i + 'EPOCHS: %4d]\t' % config.TOTAL_EPOCHS +
#                       '[step: %6d/ ' % step + 'STEPS: %6d]\t' % config.TOTAL_STEPS +
#                       '[loss:%.4f]' % AvgLoss + '/[learning rate: %6f]' % lr)
#                 AvgLoss = 0
#
#         if i % config.LOG_EPOCH == 5:
#             acc = testModel(vision_transformer)
#             if acc > BAcc:
#                 vision_transformer.save_weights(config.SAVE_PATH)
#                 BAcc = acc
#             print(f'[epoch: %4d/ ' % i + 'EPOCHS:%4d]\t' % config.TOTAL_EPOCHS +
#                   + '[acc:%.4f' % acc + ', BAcc:%.4f]' % BAcc + ' path:%s' % config.SAVE_PATH)

import tensorflow as tf
DTYPE = tf.float32

# dataset
RESIZE = 512
CROP = 384
TRAIN_PATH = '../input/cifar10/cifar/train.csv'
TEST_PATH = '.'
BATCH_SIZE = 512
SHUFFLE_BUFFER = 50000

# model
SIZE = (16, 16)
HIDDEN_SIZE = 768
MLP_DIM = 3072
NUM_HEADS = 12
NUM_LAYERS = 12
ATTENTION_DROPOUT_RATE = 0.
DROPOUT_RATE = 0.1
CLASSIFIER = 'token'
REPRESENTATION_SIZE = None
FORWARD_EXPANSION = 4
NUM_CLASSES = 10

# train
WARMUP_STEPS = 500
BASE_LR = 0.03
DECAY_TYPE = 'cosine'  # 'linear'
GRAD_NORM_CLIP = 1.
TOTAL_EPOCHS = 100
TOTAL_STEPS = TOTAL_EPOCHS*int((50000/BATCH_SIZE)-1)

# test
TEST_PATH = '../input/cifar10/cifar/test.csv'
LOAD_PATH = '../input/weights/weights/M'

# evaluate
LOG_EPOCH = 5
LOG_LOSS = 20
SAVE_PATH = './weights/M'
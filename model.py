import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers


class Encoder(layers.Layer):
    def __init__(
            self,
            num_layers,
            mlp_dim,
            inputs_positions=None,
            dropout_rate=0.1,
            train=False,
    ):
        self.num_layers = num_layers,
        self.mlp_dim = mlp_dim,
        self.inputs_positions = inputs_positions
        self.dropout_rate = dropout_rate,
        self.train = train

        self.dropout = layers.Dropout(rate=self.dropout_rate)
        self.encode_list = []
        for layer in range(self.num_layers):
            norm = layers.LayerNormalization(axis=-1)
            selfAttention = layers.Attention()



class VisionTransformer(keras.Model):
    def __init__(
            self,
            num_classes=1000,
            train=False,
            resnet=None,
            patches=None,
            hidden_size=None,
            transformer=None,
            representation_size=None,
            classifier='gap',
    ):
        self.num_classes = num_classes
        self.train = train
        self.resnet = resnet
        self.patches = patches
        self.hidden_size = hidden_size
        self.transformer = transformer
        self.representation_size = representation_size
        self.classifier = classifier

        if self.resnet is not None:
            # width = int(64*self.resnet.width_factor)
            print('no resnet code, please set resnet=None')
        self.embedding = layers.Conv2D(
            filters=self.hidden_size,
            kernel_size=self.patches,
            strides=self.patches,
            padding='valid',
            name='embedding'
        )

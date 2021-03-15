import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers


from transformerblock import TransformerBlock
import config


class Encoder(layers.Layer):
    def __init__(
            self,
            num_layers,
            mlp_dim,
            inputs_positions=None,
            dropout_rate=0.1,
    ):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.mlp_dim = mlp_dim
        self.inputs_positions = inputs_positions
        self.dropout_rate = dropout_rate

        self.dropout = layers.Dropout(rate=self.dropout_rate)
        self.encode_list = []
        for i in range(self.num_layers):
            self.encode_list.append(
                TransformerBlock(
                    mpl_dim=self.mlp_dim,
                    embed_size=config.HIDDEN_SIZE,
                    num_heads=config.NUM_HEADS,
                    forward_expansion=config.FORWARD_EXPANSION,
                    attention_dropout_rate=config.ATTENTION_DROPOUT_RATE,
                    dropout_rate=config.DROPOUT_RATE,
                    out_dim=config.HIDDEN_SIZE,
                    mask=False,
                )
            )
        self.encoded = layers.LayerNormalization(axis=-1)

    def call(self, inputs):
        #addPosemb
        pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
        pe = tf.initializers.random_normal(stddev=0.02)(shape=pos_emb_shape)
        inputs = tf.math.add(inputs, pe)

        out = self.dropout(inputs)
        for i in range(self.num_layers):
            out = self.encode_list[i](out)
        encoded = self.encoded(out)
        return encoded


class VisionTransformer(keras.Model):
    """
    for image
    input shape: NHWC
    """
    def __init__(
            self,
            num_classes=1000,
            num_layers=12,
            train=False,
            resnet=None,
            patches=None,
            hidden_size=None,
            representation_size=None,
            classifier='token',
    ):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.train = train
        self.resnet = resnet
        self.patches = patches
        self.hidden_size = hidden_size
        self.representation_size = representation_size
        self.classifier = classifier

        if self.resnet is not None:
            # width = int(64*self.resnet.width_factor)
            print('no resnet code, please set resnet=None')

        # using a conv layer to make --> B*24*24*768
        self.embedding = layers.Conv2D(
            filters=self.hidden_size,
            kernel_size=self.patches,
            strides=self.patches,
            padding='valid',
            name='embedding'
        )

        self.encoder = Encoder(
            num_layers=self.num_layers,
            mlp_dim=config.MLP_DIM,
            inputs_positions=None,
            dropout_rate=config.DROPOUT_RATE
        )

        if self.representation_size is not None:
            self.Dense = layers.Dense(representation_size)
            self.tanh = keras.activations.tanh
        self.classifier = layers.Dense(self.num_classes, kernel_initializer=keras.initializers.zeros)

    def call(self, inputs, training=None, mask=None):
        x = self.embedding(inputs)
        n, h, w, c = x.shape
        x = tf.reshape(x, shape=[n, h*w, c])
        cls = tf.zeros((1, 1, c))
        cls = tf.tile(cls, [n, 1, 1])
        x = tf.concat([cls, x], axis=1)
        x = self.encoder(x)
        x = x[:, 0]
        out = self.classifier(x)
        return out


def test():
    img = tf.random.normal((1, 384, 384, 3))
    # encoder = Encoder(num_layers=1, mlp_dim=config.MLP_DIM, inputs_positions=None, dropout_rate=0.1)
    # out = encoder(img)
    model = VisionTransformer(
        num_classes=config.NUM_CLASSES,
        num_layers=1,
        train=False,
        resnet=None,
        patches=config.SIZE,
        hidden_size=config.HIDDEN_SIZE,
        representation_size=None,
        classifier='token',)
    out = model(img)
    print(out.shape)


if __name__ == '__main__':
    test()
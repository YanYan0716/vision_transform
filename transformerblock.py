import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras as keras
import tensorflow_addons as tfa


import config


class SelfAttention(layers.Layer):
    def __init__(self, embed_size=768, heads=12, mask=False):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = self.embed_size // self.heads
        self.mask = mask

        assert (self.head_dim*heads == embed_size), "Embed size needs to be div by heads"

        self.values = layers.Dense(self.head_dim, use_bias=False)
        self.keys = layers.Dense(self.head_dim, use_bias=False)
        self.queries = layers.Dense(self.head_dim, use_bias=False)
        self.fc_out = layers.Dense(self.embed_size)

    def call(self, inputs):
        if self.mask:
            print('please set mask=False')
        else:
            values = inputs
            keys = inputs
            query = inputs

            N = tf.shape(query)[0]
            value_len, key_len, query_len = tf.shape(values)[1], tf.shape(keys)[1], tf.shape(query)[1]
            values = tf.reshape(values, shape=[N, value_len, self.heads, self.head_dim])
            keys = tf.reshape(keys, shape=[N, key_len, self.heads, self.head_dim])
            queries = tf.reshape(query, shape=[N, query_len, self.heads, self.head_dim])

            values = self.values(values)
            keys = self.keys(keys)
            queries = self.queries(queries)

            energy = tf.einsum('nqhd, nkhd->nhqk', queries, keys)

            if self.mask:
                print('please set mask False')

            attention = tf.nn.softmax(energy / (self.embed_size**(1/2)), axis=3)
            out = tf.einsum('nhql, nlhd->nqhd', attention, values)
            out = tf.reshape(out, shape=[N, query_len, self.heads*self.head_dim])
            out = self.fc_out(out)
        return out


class MlpBlock(layers.Layer):
    def __init__(self, mlp_dim, out_dim, dropout_rate=0.1):
        super(MlpBlock, self).__init__()
        self.mlp_dim = mlp_dim
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate

        self.layer1 = layers.Dense(
                self.mlp_dim,
                kernel_initializer=tf.initializers.glorot_uniform,
                bias_initializer=tf.initializers.random_normal(stddev=1e-6)
            )
        self.gelu = tfa.activations.gelu
        self.layer2 = keras.Sequential([
            layers.Dropout(rate=self.dropout_rate),
            layers.Dense(
                self.out_dim,
                kernel_initializer=tf.initializers.glorot_uniform,
                bias_initializer=tf.initializers.random_normal(stddev=1e-6)
            ),
            layers.Dropout(rate=self.dropout_rate),
        ])

    def call(self, inputs):
        out = self.layer1(inputs)
        out = self.gelu(out)
        out = self.layer2(out)
        return out


class TransformerBlock(layers.Layer):
    def __init__(
            self,
            mpl_dim=3072,
            embed_size=768,
            num_heads=12,
            forward_expansion=4,
            attention_dropout_rate=0.0,
            dropout_rate=0.0,
            out_dim=768,
            mask=False,
    ):
        super(TransformerBlock, self).__init__()
        self.mpl_dim = mpl_dim
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.forward_expansion = forward_expansion
        self.attention_dropout_rate = attention_dropout_rate
        self.dropout_rate = dropout_rate
        self.out_dim = out_dim
        self.mask = mask

        self.norm_att = layers.LayerNormalization(axis=-1)
        # self attention
        self.SelfAttention = SelfAttention(embed_size=self.embed_size, heads=self.num_heads, mask=self.mask)
        # self.feed_forward = keras.Sequential([
        #     layers.LayerNormalization(axis=-1),
        #     layers.Dropout(rate=self.attention_dropout_rate),
        #     layers.Dense(self.embed_size*4),
        #     layers.ReLU(),
        #     layers.Dense(self.embed_size),
        #     layers.LayerNormalization(axis=-1),
        #     layers.Dropout(rate=self.attention_dropout_rate)
        # ])
        self.dropout = layers.Dropout(rate=self.dropout_rate)

        # mlp
        self.norm_mlp = layers.LayerNormalization(axis=-1)
        self.Mlpblock = MlpBlock(
            mlp_dim=self.mpl_dim,
            out_dim=self.out_dim,
            dropout_rate=self.dropout_rate,
        )

    def call(self, inputs):
        # attention
        x = self.norm_att(inputs)
        attention = self.SelfAttention(x)
        attention = self.dropout(attention)
        out1 = tf.math.add(attention, x)
        # out1 = self.feed_forward(attention)
        # mlp
        out2 = self.norm_mlp(out1)
        out2 = self.Mlpblock(out2)
        out = tf.math.add(out1, out2)
        return out


def test():
    x = tf.random.normal((1, 24, 24, 768))
    n, h, w, c = x.shape
    x = tf.reshape(x, shape=[n, h*w, c])

    trans = TransformerBlock(
        mpl_dim=config.MLP_DIM,
        embed_size=config.HIDDEN_SIZE,
        num_heads=config.NUM_HEADS,
        forward_expansion=config.FORWARD_EXPANSION,
        attention_dropout_rate=config.ATTENTION_DROPOUT_RATE,
        dropout_rate=config.DROPOUT_RATE,
        out_dim=config.HIDDEN_SIZE,
        mask=False,
    )
    out =trans(x)
    print(out.shape)


if __name__ == '__main__':
    test()
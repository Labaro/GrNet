from tlinalg import *
import tensorflow as tf
from tensorflow.keras import layers
from ops import t_full_rank_mapping


class FRMap(layers.Layer):
    def __init__(self, filter, output_dim, **kwargs):
        super(FRMap, self).__init__(**kwargs)
        self.filter = filter
        self.output_dim = output_dim

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(self.filter, self.output_dim, input_shape[1]),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, inputs, **kwargs):
        outputs = tf.transpose(t_full_rank_mapping(inputs, self.W), (2, 1, 0, 3))
        return outputs
        # return tf.transpose(tf.tensordot(self.W, inputs, [[-1], [1]]), (2, 1, 0, 3))


class ReOrth(layers.Layer):
    def __init__(self, **kwargs):
        super(ReOrth, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        inputs = tf.cast(inputs, tf.complex64)
        fft_inputs = tf.signal.fft(inputs)
        fft_inputs_t = tf.transpose(fft_inputs, (0, 3, 1, 2))
        S, U, V = tf.linalg.svd(fft_inputs_t)
        outputs = tf.signal.ifft(tf.transpose(U, (0, 2, 3, 1)))
        return tf.math.real(outputs)


class ProjMap(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        inputs = tf.cast(inputs, tf.complex64)
        fft_inputs = tf.transpose(tf.signal.fft(inputs), (0, 3, 1, 2))
        fft_inputs_t = tf.transpose(fft_inputs, (0, 1, 3, 2))
        fft_outputs = tf.matmul(fft_inputs, fft_inputs_t)
        return tf.math.real(tf.transpose(fft_outputs, (0, 2, 3, 1)))


class ProjPooling(layers.Layer):
    def __init__(self, pool_size=4, **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size

    def call(self, inputs, **kwargs):
        # Try tf.layers.Permute((2,3))
        # with tf.device("/cpu:0"):
        inputs_t = tf.transpose(inputs, (0, 1, 3, 2))
        outputs_t = tf.nn.avg_pool(inputs_t, ksize=[1, 1, self.pool_size, 1], strides=[1, 1, self.pool_size, 1],
                                   padding="SAME", data_format="NHWC")
        # return outputs_t
        return tf.transpose(outputs_t, (0, 1, 3, 2))


class OrthMap(layers.Layer):
    def __init__(self, nb_eigen, **kwargs):
        super().__init__(**kwargs)
        self.nb_eigen = nb_eigen

    def call(self, inputs, **kwargs):
        inputs = tf.cast(inputs, tf.complex64)
        fft_inputs = tf.signal.fft(inputs)
        fft_inputs_t = tf.transpose(fft_inputs, (0, 3, 1, 2))
        S, U, V = tf.linalg.svd(fft_inputs_t)
        outputs = tf.signal.ifft(tf.transpose(U[:, :, :, :self.nb_eigen], (0, 2, 3, 1)))
        return tf.math.real(outputs)

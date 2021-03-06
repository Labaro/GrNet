from tlinalg import *
import tensorflow as tf
from tensorflow.keras import layers
from ops import t_full_rank_mapping, t_re_orthonormalization, t_orthonormal_mapping


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
        temp = tf.concat(tf.unstack(t_full_rank_mapping(inputs, self.W), axis=3), axis=0)
        outputs = tf.transpose(temp, (2, 1, 0, 3))
        return outputs
        # return tf.transpose(tf.tensordot(self.W, inputs, [[-1], [1]]), (2, 1, 0, 3))


class ReOrth(layers.Layer):
    def __init__(self, **kwargs):
        super(ReOrth, self).__init__(**kwargs)

    def build(self, input_shape):
        self.eye = tf.eye(input_shape[1], input_shape[1], batch_shape=[input_shape[3]], dtype=tf.complex64)

    def call(self, inputs, **kwargs):
        inputs = tf.cast(inputs, tf.complex64)
        fft_inputs = tf.signal.fft(inputs)
        fft_inputs_t = tf.transpose(fft_inputs, (0, 3, 1, 2))
        q = t_re_orthonormalization(fft_inputs_t, self.eye)
        outputs = tf.signal.ifft(tf.transpose(q, (0, 2, 3, 1)))
        return tf.math.real(outputs)


class ProjMap(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        inputs = tf.cast(inputs, tf.complex64)
        inputs_t = tf.transpose(inputs, (0, 2, 1, 3))
        fft_inputs = tf.transpose(tf.signal.fft(inputs), (0, 3, 1, 2))
        fft_inputs_t = tf.transpose(tf.math.conj(tf.signal.fft(inputs_t)), (0, 3, 1, 2))
        fft_outputs = tf.matmul(fft_inputs, fft_inputs_t)
        return tf.math.real(tf.signal.ifft(tf.transpose(fft_outputs, (0, 2, 3, 1))))


class ProjPooling(layers.Layer):
    def __init__(self, pool_size=4, **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size

    def call(self, inputs, **kwargs):
        inputs_t = tf.transpose(inputs, (0, 1, 3, 2))
        outputs_t = tf.nn.avg_pool(inputs_t, ksize=[1, 1, self.pool_size, 1], strides=[1, 1, self.pool_size, 1],
                                   padding="SAME", data_format="NHWC")
        return tf.transpose(outputs_t, (0, 1, 3, 2))


class OrthMap(layers.Layer):
    def __init__(self, nb_eigen, **kwargs):
        super().__init__(**kwargs)
        self.nb_eigen = nb_eigen

    def build(self, input_shape):
        self.zeros = tf.zeros((1, input_shape[3], input_shape[1], input_shape[2] - self.nb_eigen), dtype=tf.complex64)

    def call(self, inputs, **kwargs):
        inputs = tf.cast(inputs, tf.complex64)
        fft_inputs = tf.signal.fft(inputs)
        fft_inputs_t = tf.transpose(fft_inputs, (0, 3, 1, 2))
        # s,u,v = tf.linalg.svd(fft_inputs_t)
        u = t_orthonormal_mapping(fft_inputs_t, self.nb_eigen, self.zeros)
        u = tf.transpose(u, (0, 2, 3, 1))
        outputs = tf.signal.ifft(u)
        return tf.math.real(outputs)

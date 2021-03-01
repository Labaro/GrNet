from tlinalg import *
import tensorflow as tf
from tensorflow.keras import layers


def full_rank_mapping(X, W):
    """
    :param X: batch data of dimension (n, d, q) and n the size of the batch.
    :param W: batch filters of dimension (m, dk, d) where dk < d.
    :return: the full rank product W.X of dimension (n * m, dk, q)
    """
    return np.dot(W, X).transpose((0, 2, 1, 3)).reshape((X.shape[0] * W.shape[0], W.shape[1], X.shape[2]))


@tf.function
def t_full_rank_mapping(X, W):
    """
    :param X: batch data of dimension (n, d0, q) where n is the size of the batch
    :param W: batch filters of dimension (m, d1, d0), a list of matrices. Each filter is applied on each slice of the
     tensor X
    :return: tensors batch of dimension (n, d1, m, q) after concatenating every results of dimension (d1, q) m times
    """
    return np.stack([np.stack([np.dot(w, x) for w in W], axis=1) for x in X])


def re_orthonormalization(X):
    """
    :param X: batch data of dimension (n, d, q)
    :return: a batch Q of dimension (n, d, q)
    """
    # np.linalg.qr return Q, R with Q square and R not square but torch.qr return Q not square and R square as in the
    # paper


@tf.function
def t_re_orthonormalization(X, mode="econ", l=None):
    """
    :param X: tensors batch of dimension (n, d1, m, q)
    :return: tensors batch U of dimension (n, d1, m, d1)
    """
    if l is not None:
        return np.stack([t_svd(x, opt=mode)[0][:, :l, :] for x in X])
    else:
        return np.stack([t_svd(x, opt=mode)[0] for x in X])


def projection_mapping(X):
    """
    :param X: a batch of dimension (n, d, q)
    :return: a batch of dimension (n, d, d)
    """
    return np.dot(X, X.T)


def t_projection_mapping(X):
    """
    :param X: tensors batch of dimension (n, d1, m, d1)
    :return: tensors batch of dimension (n, d1,
    """
    return np.stack([t_product(x, t_transpose(x)) for x in X])


def projection_pooling(X, pool_size=4):
    """
    :param X: a batch of dimension (n, d, q)
    :param pool_size: int, number of matrices to pool together: partial mean is computed on pool_size number of matrices
    :return: a batch data of dimension (n / pool_size, d, q)
    """
    return


def t_projection_pooling(X, pool_size=2):
    """
    :param X: a tensor of dimension (n, d1, d1, q)
    :param pool_size: int, number of slices to pool together: partial mean is computed on pool_size number of slices
    :return: a tensor of dimension (d, n / pool_size, q)
    """
    split_idx = np.arange(pool_size, X.shape[3], pool_size)
    sub_array_list = np.split(X, split_idx, axis=-1)
    if X.shape[3] % pool_size == 0:
        split_array = np.stack(sub_array_list, axis=-2)
        mean_tensor = np.mean(split_array, axis=-1)
    else:
        partial_array_1 = np.stack(sub_array_list[:-1], axis=-2)
        partial_array_2 = np.stack(sub_array_list[-1:], axis=-2)
        partial_mean_1 = np.mean(partial_array_1, axis=-1)
        partial_mean_2 = np.mean(partial_array_2, axis=-1)
        mean_tensor = np.concatenate([partial_mean_1, partial_mean_2], axis=-1)
    return mean_tensor


def orthonormal_mapping(X, l):
    """
    :param X: a batch of dimension (n, d, q)
    :param l: int, number of eigenvectors to be conserved. l have to be less than min(d, q)
    :return: a batch data of dimension (n, l, l)
    """
    return


def t_orthonormal_mapping(X, l):
    """
    :param X: a tensor of dimension (d, n, q)
    :param l: number of eigen-matrix to keep.
    :return: U the tensor of the l biggest eigen-matrices of dimension
    """
    rank = tubal_rank(X)
    if l > rank:
        l = rank  # TODO: This should be corrected. If we ask l eigen-matrices, this should return l eigen-matrices. Not less. But the complete SVD is very time-consuming.
    U, S, V = t_svd(X)
    return U[:, :l, :]


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
        return tf.transpose(tf.tensordot(self.W, inputs, [[-1], [1]]), (2, 1, 0, 3))


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
        outputs_t = tf.nn.max_pool(inputs_t, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding="SAME", data_format="NHWC")
        #return outputs_t
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

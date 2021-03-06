import tensorflow as tf
import numpy as np


@tf.custom_gradient
def custom_ops(inputs):
    # Do whatever we want
    def grad(upstream):
        # Compute gradient
        return

    outputs = None
    return outputs, grad


@tf.function
@tf.custom_gradient
def t_full_rank_mapping(inputs, weights):
    """
    Compute the full rank mapping between the inputs and the weights that are linear applications.
    :param inputs: (None, d0, q) batch of images of dimension (d0, q)
    :param weights: (m, d1, d0) each filter is a matrix of dimension (d1, d0)
    :return: a batch of tensor of size (m, d1, None, q)
    """

    def grad(upstream):
        """
        Custom gradient to have the update in the tangent space.
        :param upstream: dL^{k+1}/dX_k of size (m, d1, None, q)
        :return: None, gradL^k
        """
        euc_grad = tf.tensordot(upstream, inputs, [[2, 3, 4], [0, 2, 3]])
        projected_grad = euc_grad - tf.matmul(tf.matmul(euc_grad, weights, transpose_b=True), weights)
        return None, projected_grad

    outputs = tf.tensordot(weights, inputs, [[-1], [1]])
    return outputs, grad


@tf.function
@tf.custom_gradient
def t_re_orthonormalization(inputs, eye):
    print(inputs)
    q, r = tf.linalg.qr(inputs, full_matrices=False)
    print(q,r)

    def grad(upstream):
        s = eye - tf.matmul(q, q, transpose_b=True)
        term_1 = tf.matmul(s, upstream, transpose_a=True)
        temp_term = tf.matmul(q, upstream, transpose_a=True)
        term_2 = tf.matmul(q, tf.experimental.numpy.tril(temp_term) - tf.experimental.numpy.tril(
            tf.transpose(temp_term, (0, 1, 3, 2))))
        dL_dX = tf.matmul(term_1 + term_2, tf.transpose(tf.linalg.inv(r), (0, 1, 3, 2)))
        return dL_dX, None

    return q, grad


@tf.function
@tf.custom_gradient
def t_orthonormal_mapping(inputs, nb_eigen, zeros):
    s, u, v = tf.linalg.svd(inputs)

    def grad(upstream):
        bc_zeros = tf.repeat(zeros, tf.shape(upstream)[0], axis=0)
        dL_du = tf.concat([upstream, bc_zeros], axis=-1)
        temp = tf.broadcast_to(tf.stack([s], axis=-1), tf.shape(u))
        temp = temp - tf.transpose(temp, (0, 1, 3, 2))
        K = tf.cast(tf.math.divide_no_nan(tf.ones(tf.shape(u)), temp), tf.complex64)
        dL_dX = tf.matmul(tf.matmul(u, tf.multiply(K, tf.matmul(v, dL_du, transpose_a=True))), v, transpose_b=True)
        return dL_dX, None, None

    return u[:, :, :, :nb_eigen], grad

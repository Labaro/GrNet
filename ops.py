import tensorflow as tf


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
        euc_grad = tf.tensordot(upstream, inputs, [[2, 3], [0, 2]])
        projected_grad = euc_grad - tf.matmul(tf.matmul(euc_grad, weights, transpose_b=True), weights)
        return None, projected_grad

    outputs = tf.tensordot(weights, inputs, [[3], [1]])
    return outputs, grad

@tf.function
@tf.custom_gradient
def t_re_orthonormalization(inputs):
    def grad(upstream):
        return

    outputs = None
    return outputs, grad

@tf.function
@tf.custom_gradient
def t_orthonormal_mapping(inputs):
    def grad(upstream):
        return

    outputs = None
    return outputs, grad

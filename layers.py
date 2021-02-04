from tlinalg import *


def full_rank_mapping(X, W):
    """
    :param X: batch data of dimension (n, d, q) and n the size of the batch.
    :param W: batch filters of dimension (m, dk, d) where dk < d.
    :return: the full rank product W.X of dimension (n * m, dk, q)
    """
    return np.dot(W, X).transpose((0, 2, 1, 3)).reshape((X.shape[0] * W.shape[0], W.shape[1], X.shape[2]))


def t_full_rank_mapping(X, W):
    """
    :param X: batch data of dimension (d, n, q) where n is the size of the batch
    :param W: batch filters of dimension:
     Option 1 : (m, dk, d, q), (dk, d, q) is the dimension of tensor filter. And we have m filters in total
     Option 2 : (dk, d, q), a single tensor but the second dimension has to be the same as the first dimension of X
     Option 3 : (m, dk, d), a list of matrices. Each filter is applied on each slice of the tensor X
    :return:
     Option 1 : (dk, n * m, q) after concatenating every results of dimension (dk, n, q) m times
     Option 2 : (dk, n, q)
     Option 3 : (dk, n * m, q) -> retenue en faisant:
     for i in 0:n, for j in 0:m, repeat:
     X[:, i, :] -> dim = (d, 1, q) -> reshape dim = (d, q, 1)
     W[j, :, :] -> dim = (1, dk, d) -> reshape dim = (dk, d, 1)
      => t_product(W[j, :, :], X[:,i,:]), dim = (dk, q, 1) -> reshape (dk, 1, q)
    After the loop, dim = (dk, n * m, q)
    """
    # Option 1: W is a batch of tensors:
    # return np.concatenate([t_product(W[i], X) for i in range(W.shape[0])], axis=1)
    # Option2: W is a single tensor:
    # return t_product(W, X)
    # Option 3: W is a list of matrices:
    return np.concatenate([np.transpose(
        t_product(W[j, :, :].reshape((W.shape[1], W.shape[2], 1)), X[:, i, :].reshape((X.shape[0], X.shape[2], 1))),
        (0, 2, 1)) for i in range(X.shape[1]) for j in range(W.shape[0])], axis=1)


def re_orthonormalization(X):
    """
    :param X: batch data of dimension (n, d, q)
    :return: a batch Q of dimension (n, d, q)
    """
    # np.linalg.qr return Q, R with Q square and R not square but torch.qr return Q not square and R square as in the
    # paper


def t_re_orthonormalization(X, mode="reduced"):
    """
    :param X: tensor of dimension (d, n, q)
    :return: a tensor Q of dimension (d, d, q) where X = t_product(Q, R)
    """
    Q, R = t_qr(X, mode=mode)
    return Q


def projection_mapping(X):
    """
    :param X: a batch of dimension (n, d, q)
    :return: a batch of dimension (n, d, d)
    """
    return np.dot(X, X.T)


def t_projection_mapping(X):
    """
    :param X: a tensor of dimension (d, n, q)
    :return: a tensor of dimension (d, d, q) ???? Should have (d, n, d)?
    """
    return t_product(X, t_transpose(X))


def projection_pooling(X, pool_size=4):
    """
    :param X: a batch of dimension (n, d, q)
    :param pool_size: int, number of matrices to pool together: partial mean is computed on pool_size number of matrices
    :return: a batch data of dimension (n / pool_size, d, q)
    """
    return


def t_projection_pooling(X, pool_size=4):
    """
    :param X: a tensor of dimension (d, n, q)
    :param pool_size: int, number of slices to pool together: partial mean is computed on pool_size number of slices
    :return: a tensor of dimension (d, n / pool_size, q)
    """
    n, t = X.shape[1] % pool_size, X.shape[1] // pool_size
    return np.hstack([np.mean(np.stack(np.split(X[:, :-n, :], t, axis=1), axis=1), axis=2),
                      np.mean(X[:, -n:, :], axis=1).reshape((X.shape[0], 1, X.shape[2]))])


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

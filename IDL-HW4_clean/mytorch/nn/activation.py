import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")

        Z_stable = Z - np.max(Z, axis=self.dim, keepdims=True)
        self.A = np.exp(Z_stable) / np.sum(np.exp(Z_stable), axis=self.dim, keepdims=True)

        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # 1. Identify the dimension along which softmax was applied in the forward pass:
        # self.dim
        # 2. Move dimension self.dim to the last position to obtain a tensor of shape (..., ..., C)
        dLdA_mov_dim = np.moveaxis(dLdA, self.dim, -1)
        A_mov_dim = np.moveaxis(self.A, self.dim, -1)
        # 3. Flatten the remaining dimensions to obtain a 2D tensor of shape (*,C)
        dLdA_flatten = dLdA_mov_dim.reshape(-1, dLdA_mov_dim.shape[-1])
        A_flatten = A_mov_dim.reshape(-1, A_mov_dim.shape[-1])
        # 4. Perform operations on this 2D tensor as done in HW1P1
        N = A_flatten.shape[0]
        C = A_flatten.shape[1]

        dLdZ = np.zeros([N, C])

        for i in range(N):
            J = np.zeros([C, C])
            for m in range(C):
                for n in range(C):
                    J[m, n] = np.where(m == n, A_flatten[i, m] * (1 - A_flatten[i, m]), -A_flatten[i, m] * A_flatten[i, n]).item()
            dLdZ[i, :] = dLdA_flatten[i, :] @ J

        # 5. Reshape the tensor back to the nD shape (..., ...,C)
        dLdZ = dLdZ.reshape(dLdA_mov_dim.shape)
        # 6. Finally, move the dim dimension back to its original position to revert back to the original shape
        return np.moveaxis(dLdZ, -1, self.dim)



    
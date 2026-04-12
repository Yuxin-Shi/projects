import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        """
        Initialize the weights and biases with zeros
        W shape: (out_features, in_features)
        b shape: (out_features,)  # Changed from (out_features, 1) to match PyTorch
        """
        # DO NOT MODIFY
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros(out_features)


    def init_weights(self, W, b):
        """
        Initialize the weights and biases with the given values.
        """
        # DO NOT MODIFY
        self.W = W
        self.b = b

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (*, in_features)
        :return: Output Z with shape (*, out_features)
        
        Handles arbitrary batch dimensions like PyTorch
        """
        self.input_shape = A.shape
        in_features = self.input_shape[-1]
        A_flatten = A.reshape(-1, in_features)

        Z = A_flatten @ self.W.T + self.b
        Z = Z.reshape(self.input_shape[:-1] + (Z.shape[-1],))
        
        # Store input for backward pass
        self.A = A
        
        return Z

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (*, out_features)
        :return: Gradient of loss wrt input A (*, in_features)
        """
        out_features = dLdZ.shape[-1]
        dLdZ_flatten = dLdZ.reshape(-1, out_features)
        A_faltten = self.A.reshape(-1, self.A.shape[-1])

        # Compute gradients
        self.dLdA = dLdZ_flatten @ self.W
        self.dLdW = dLdZ_flatten.T @ A_faltten
        self.dLdb = np.sum(dLdZ_flatten, axis=0)

        self.dLdA = self.dLdA.reshape(self.A.shape)

        # Return gradient of loss wrt input
        return self.dLdA

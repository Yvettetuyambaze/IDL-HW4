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
        # Store input for backward pass
        self.A = A
        # Store original shape for later unflattening
        self.original_shape = A.shape
        # Flatten all dimensions except the last one (in_features)
        A_flat = A.reshape(-1, A.shape[-1])
        # Compute Z = A W^T + b
        Z_flat = A_flat @ self.W.T + self.b
        # Reshape back to original dimensions except last dim replaced by out_features
        new_shape = self.original_shape[:-1] + (self.W.shape[0],)
        Z = Z_flat.reshape(new_shape)
        return Z

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (*, out_features)
        :return: Gradient of loss wrt input A (*, in_features)
        """
        # Flatten dLdZ and A to 2D
        dLdZ_flat = dLdZ.reshape(-1, dLdZ.shape[-1])
        A_flat = self.A.reshape(-1, self.A.shape[-1])

        # Compute gradients
        # dLdA = dLdZ @ W
        self.dLdA = (dLdZ_flat @ self.W).reshape(self.original_shape)
        # dLdW = (dLdZ)^T @ A
        self.dLdW = dLdZ_flat.T @ A_flat
        # dLdb = sum over batch dimension of dLdZ
        self.dLdb = dLdZ_flat.sum(axis=0)

        # Return gradient of loss wrt input
        return self.dLdA
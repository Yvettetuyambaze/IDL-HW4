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
        
        # Store original shape and dimension
        self.original_shape = Z.shape
        self.dim = self.dim if self.dim >= 0 else len(Z.shape) + self.dim

        # Move the target dimension to the end, flatten others
        Z_permuted = np.moveaxis(Z, self.dim, -1)
        Z_flat = Z_permuted.reshape(-1, Z_permuted.shape[-1])

        # Numerically stable softmax
        max_vals = np.max(Z_flat, axis=1, keepdims=True)
        exp_Z = np.exp(Z_flat - max_vals)
        A_flat = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

        # Reshape back and move dimension to original position
        A_reshaped = A_flat.reshape(Z_permuted.shape)
        self.A = np.moveaxis(A_reshaped, -1, self.dim)

        # Store for backward
        self.Z_flat = Z_flat
        self.A_flat = A_flat
        self.Z_permuted_shape = Z_permuted.shape
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # Move dim to end and flatten dLdA similarly
        dLdA_permuted = np.moveaxis(dLdA, self.dim, -1)
        dLdA_flat = dLdA_permuted.reshape(-1, dLdA_permuted.shape[-1])

        # For each row, compute Jacobian-vector product
        # J = diag(a) - a a^T; dL/dz = dL/da @ J = dL/da * diag(a) - (dL/da * a^T) a
        # = a * dL/da - a * (dL/da dot a)
        # More efficiently: dL/dz = a * (dL/da - (dL/da * a).sum(axis=1, keepdims=True))
        dLdZ_flat = self.A_flat * (dLdA_flat - np.sum(dLdA_flat * self.A_flat, axis=1, keepdims=True))

        # Reshape and move dimension back
        dLdZ_reshaped = dLdZ_flat.reshape(self.Z_permuted_shape)
        dLdZ = np.moveaxis(dLdZ_reshaped, -1, self.dim)
        return dLdZ
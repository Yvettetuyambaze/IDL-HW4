import numpy as np
from .activation import Softmax

class ScaledDotProductAttention:
    """
    Scaled Dot Product Attention
    """ 
    def __init__(self):
        '''
        Initialize the ScaledDotProductAttention class.
        '''
        # Initialize your softmax layer
        # What dimension should you pass to the softmax constructor?
        self.eps = 1e10 # DO NOT MODIFY
        self.softmax = Softmax(dim=-1)  # softmax over last dimension (key positions)
        
    
    def forward(self, Q, K, V, mask=None):
        """
        :param Q: Query matrix of shape (N, ..., H, L, E) where L is target sequence length
        :param K: Key matrix of shape (N, ..., H, S, E) where S is source sequence length
        :param V: Value matrix of shape (N, ..., H, S, Ev) where Ev is value dimension
        :param mask: Boolean mask matrix of shape (N, ..., H, L, S) or broadcastable shape where 1/True indicates a position to ignore
        :return: Output matrix of shape (N, ..., H, L, Ev)
        """
        # Store inputs for backward
        self.Q = Q
        self.K = K
        self.V = V
        self.mask = mask

        # Calculate attention scores
        d_k = Q.shape[-1]
        # Use matmul for the last two dimensions: (..., L, E) @ (..., E, S) -> (..., L, S)
        # But Q and K have shape (..., L, E) and (..., S, E). We need K transposed.
        scaled_dot_product = np.matmul(Q, np.swapaxes(K, -1, -2)) / np.sqrt(d_k)
        self.scaled_dot_product = scaled_dot_product

        # Apply mask before softmax if provided
        if mask is not None:
            scaled_dot_product = np.where(mask, -self.eps, scaled_dot_product)

        # Compute attention scores: softmax over last dimension (S)
        self.attention_scores = self.softmax.forward(scaled_dot_product)

        # Calculate final output: (..., L, S) @ (..., S, Ev) -> (..., L, Ev)
        output = np.matmul(self.attention_scores, V)

        return output
    
    def backward(self, d_output):
        """
        :param d_output: Gradient of loss wrt output of shape (N, ..., H, L, Ev)
        :return: Gradient of loss wrt input Q, K, V
        """
        # dV = A^T @ dO  (shape: (..., S, Ev))
        dV = np.matmul(np.swapaxes(self.attention_scores, -1, -2), d_output)

        # dA = dO @ V^T   (shape: (..., L, S))
        dA = np.matmul(d_output, np.swapaxes(self.V, -1, -2))

        # dS = softmax_backward(dA)
        dS = self.softmax.backward(dA)

        # Apply mask gradient (where mask is True, gradient is zero)
        if self.mask is not None:
            dS = np.where(self.mask, 0.0, dS)

        # Scale by 1/sqrt(d_k)
        d_k = self.Q.shape[-1]
        dS_scaled = dS / np.sqrt(d_k)

        # dQ = dS_scaled @ K   (shape: (..., L, E))
        dQ = np.matmul(dS_scaled, self.K)

        # dK = dS_scaled^T @ Q (shape: (..., S, E))
        dK = np.matmul(np.swapaxes(dS_scaled, -1, -2), self.Q)

        return dQ, dK, dV

import numpy as np
import torch

from model.sparse.random_projection import RandomProjection

class SparseRandomProjection(RandomProjection):
    def __init__(self, input_dim, output_dim, requires_grad=False, density=0.1, binary=False):
        """
        Initializes a sparse random encoder. The input is encoded with abs. top-k
        elements retained, and others set to zero.
        
        Args:
            input_dim (int): Number of input dimensions.
            output_dim (int): Number of output dimensions.
            density (float): Fraction of non-zero components in the projected result.
        """
        super().__init__(input_dim, output_dim, requires_grad)
        self.density = density
        self.binary = binary

    def transform(self, x):
        """
        Applies the sparse random projection to the input data.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Transformed tensor of shape (batch_size, output_dim).
        """
        y = super().transform(x)
        k = self.get_k(y)

        # Compute top-k indices based on magnitude (absolute value)
        values, indices = torch.topk(y.abs(), k, dim=1)  # top-k over dim 1

        # Create a mask to retain only the top-k values (keep their original signs)
        mask = torch.zeros_like(y)
        mask.scatter_(1, indices, 1)  # In dim 1, Scatter 1 into the positions of top-k indices
        sparse_output = y * mask

        if self.binary:
            sparse_output = SparseRandomProjection.binarize(sparse_output)
        return sparse_output

    @staticmethod
    def binarize(x):
        y = (torch.abs(x) > 0).float()
        return y
            
    @staticmethod    
    def get_volume(x) -> int:
        """
        Returns the volume of the provided tensor.
        """
        volume = np.prod(x.size())
        return volume

    def get_k(self, x) -> int:
        """
        Return the target k value for a given tensor, given the target density.
        """
        volume = np.prod(x.shape[1:])#x.size())
        k = int(float(volume) * self.density)
        return k

    @staticmethod    
    def get_density(x) -> float:
        """
        Measures the frequency of nonzero values in the provided tensor.
        """
        volume = SparseRandomProjection.get_volume(x)
        n = torch.nonzero(x).size(0)
        density = float(n) / float(volume)
        return density

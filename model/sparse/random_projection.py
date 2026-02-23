
import torch

class RandomProjection:
    def __init__(self, input_dim, output_dim, requires_grad=False):
        """
        Initialize the Random Projection with fixed random matrix.

        Args:
            input_dim (int): The dimensionality of the input space.
            output_dim (int): The dimensionality of the projected space.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Create a random projection matrix with a normal (Gaussian) distribution
        self.projection_matrix = torch.randn(
            output_dim, 
            input_dim,
            requires_grad = requires_grad,
        ) / torch.sqrt(
            torch.tensor(
                output_dim, 
                dtype=torch.float32,
                requires_grad = requires_grad,
            )
        )

    def transform(self, x):
        """
        Perform random projection on input x.

        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Projected tensor of shape (batch_size, output_dim).
        """
        # Perform the projection using matrix multiplication
        return x @ self.projection_matrix.T

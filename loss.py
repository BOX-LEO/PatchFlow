
import torch
from torch import Tensor, nn


class Loss(nn.Module):

    def forward(self, z_dist: Tensor, jacobians: Tensor) -> Tensor:
        """Loss function.

        Args:
            z_dist (Tensor): Latent space image mappings from NF.
            jacobians (Tensor): Jacobians of the distribution

        Returns:
            Loss value
        """
        z_dist = z_dist.reshape(z_dist.shape[0], -1)
        return torch.mean(0.5 * torch.sum(z_dist**2, dim=(1,)) - jacobians) / z_dist.shape[1]
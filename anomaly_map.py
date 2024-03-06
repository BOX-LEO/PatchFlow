from __future__ import annotations
import torch
import torch.nn.functional as F
from omegaconf import ListConfig
from torch import Tensor, nn


class AnomalyMap(nn.Module):
    """Generate Anomaly Heatmap."""

    def __init__(self, input_size: ListConfig | tuple) -> None:
        super().__init__()
        self.input_size = input_size if isinstance(input_size, tuple) else tuple(input_size)
        self.crop_size = tuple([int(self.input_size[0] * 0.875), int(self.input_size[1] * 0.875)])


    def forward(self, hidden_variables: Tensor) -> Tensor:
        """Generate Anomaly Heatmap.


        Args:
            hidden_variables (list[Tensor]): List of hidden variables from each NF FastFlow block.

        Returns:
            Tensor: Anomaly Map.
        """

        log_prob = -torch.mean(hidden_variables ** 2, dim=1, keepdim=True) * 0.5
        prob = torch.exp(log_prob)
        anomaly_map = F.interpolate(
            input=-prob,
            size=self.input_size,
            mode="bilinear",
            align_corners=True,
        )

        return anomaly_map

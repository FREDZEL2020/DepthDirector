import torch
import torch.nn as nn


class CamVidEncoder(nn.Module):
    """
    A VAE model for encoding camera information and video features.
    """

    def __init__(
        self,
        in_channels: int = 16,
        hidden_channels: int = 1024,
        out_channels: int = 5120,
        in_video: int=2
    ) -> None:
        super().__init__()
        self.in_video = in_video
        self.in_channels = in_channels
        self.latent_encoder = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels * in_video, hidden_channels, kernel_size=1, stride=1, padding=0),
            torch.nn.SiLU(),
            torch.nn.Conv3d(hidden_channels, hidden_channels, kernel_size=1, stride=1, padding=0),
            torch.nn.SiLU(),
            torch.nn.Conv3d(hidden_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        )
        self.latent_patch_embedding = torch.nn.Conv3d(hidden_channels, out_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        nn.init.zeros_(self.latent_patch_embedding.weight)
        nn.init.zeros_(self.latent_patch_embedding.bias)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, nn.Module):
            module.gradient_checkpointing = value

    def forward(self, latent) -> torch.Tensor:
        assert latent.shape[1] == self.in_video * self.in_channels
        latent = self.latent_encoder(latent)
        latent = self.latent_patch_embedding(latent)
        return latent

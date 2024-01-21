import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_encoder(nn.sequential):

    def __inti__(self):
        super().__inti__(
            # Reduces the dimensionality of the data but increases number of features
            # Data dimensions: (BatchSize, C, H, W) -> (B, 128, H, W)
            nn.Conv2d(3, 128, kernel_size = 3, padding = 1), 
            
            # Combinations of Convolutions and Normalizations
            VAE_ResidualBlock(128, 128), # in_channels, out_channels
            VAE_ResidualBlock(128, 128),

            # (B, 128, H, W) -> (B, 128, H/2, W/2)
            nn.Conv2d(128, 128, kernel_size = 3, stride = 2, padding = 0), 

            # (B, 128, H/2, W/2) -> (B, 256, H/2, W/2)
            VAE_ResidualBlock(128, 256),
            VAE_ResidualBlock(256, 256),

            # (B, 128, H/2, W/2) -> (B, 256, H/4, W/4)
            nn.Conv2d(256, 256, kernel_size = 3, stride = 2, padding = 0),

            # (B, 256, H/4, W/4) -> (B, 512, H/4, W/4)
            VAE_ResidualBlock(256, 512),
            VAE_ResidualBlock(512, 512),

            # (B, 512, H/4, W/4) -> (B, 512, H/8, H/8)
            nn.Conv2d(512, 512, kernel_size = 3, stride = 2, padding = 0),

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # Self-Attention Block
            # Each pixel attends with all the pixels present in the data.
            # Read the readme for the reason.
            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512, 512),

            # Group Normalization
            nn.GroupNorm(32, 512),
            
            # Sigmoid Linear Unit 
            nn.SiLU(),
            
            # (B, 512, H/8, W/8) -> (B, 8, H/8, W/8)
            nn.Conv2d(512, 8, kernel_size=3, padding = 1),

            # (B, 8, H/8, W/8) -> (B, 8, H/8, W/8)
            nn.Conv2d(8, 8, kernel_size = 1, padding = 0)
        )


    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:

        # x: (B, C, H, W)
        # noise: (B, out_channels, H/8, W/8)
        

        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                # applying asymmetrical padding
                # (pad_left, pad_right, pad_top, pad_bottom)
                x = F.pad(x, (0, 1, 0, 1))
            
            x = module(x)

        # (B, 8, H/8, W/8) => return 2 tensors of size (B, 4, H/8, W/8)
        mean, log_variance = torch.chunk(x, 2, dim = 1)

        # Before converting log_variace into variance clamp the values of log_variance in acceptable range
        log_variance = torch.clamp(log_variance, -30, 20)

        variance = log_variance.exp()

        stddev = variance.sqrt()

        # How to sample from distribution N(mean, variance).
        # By converting isotropic gaussian distribution into this distribution.
        
        # normal noise (Z) = N(0, 1) -> X = N(mean, variance)
        # X = mean + stddev * Z

        x = mean + stddev * noise

        # scale the latent space
        x *= 0.18215

        return x
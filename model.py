"""
PINO-ResNet Hybrid Model for Heston Option Pricing
Combines Fourier Neural Operators with Dense Skip Connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
import numpy as np


class SpectralConv3d(nn.Module):
    """
    3D Spectral Convolution Layer
    Performs convolution in Fourier space for efficient global operations
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 modes1: int, modes2: int, modes3: int):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            modes1, modes2, modes3: Number of Fourier modes to keep in each dimension
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        
        # Scale factor for Xavier initialization
        scale = 1 / (in_channels * out_channels)
        
        # Learnable Fourier coefficients (complex-valued)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, 2)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, 2)
        )
        self.weights3 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, 2)
        )
        self.weights4 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, 2)
        )
    
    def compl_mul3d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Complex multiplication in Fourier space.

        - `input` shape: [batch, in_channels, x, y, z, 2]  (real, imag)
        - `weights` shape: [in_channels, out_channels, x, y, z, 2]  (real, imag)
        Convert real/imag pairs to complex dtype, do einsum on complex tensors,
        then split back to real/imag for downstream code.
        """
        # Convert to complex tensors
        input_c = torch.complex(input[..., 0], input[..., 1])   # [b, i, x, y, z]
        weights_c = torch.complex(weights[..., 0], weights[..., 1])  # [i, o, x, y, z]

        # Perform complex einsum over 5 dims -> result is complex with shape [b, o, x, y, z]
        out_c = torch.einsum("bixyz,ioxyz->boxyz", input_c, weights_c)

        # Return stacked real/imag
        out = torch.stack([out_c.real, out_c.imag], dim=-1)  # [b, o, x, y, z, 2]
        return out

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, in_channels, n_s, n_v, n_t]
        Returns:
            [batch, out_channels, n_s, n_v, n_t]
        """
        batchsize = x.shape[0]
        
        # Transform to Fourier space
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        x_ft = torch.stack([x_ft.real, x_ft.imag], dim=-1)
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), 
                            x.size(-1)//2 + 1, 2, device=x.device)
        
        # Lower modes in all dimensions
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(
                x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], 
                self.weights1
            )
        
        # Higher modes in dimension 1
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(
                x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], 
                self.weights2
            )
        
        # Higher modes in dimension 2
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(
                x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], 
                self.weights3
            )
        
        # Higher modes in both dimensions 1 and 2
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(
                x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], 
                self.weights4
            )
        
        # Convert back to complex tensor
        out_ft_complex = torch.complex(out_ft[..., 0], out_ft[..., 1])
        
        # Transform back to spatial domain
        x = torch.fft.irfftn(out_ft_complex, s=(x.size(-3), x.size(-2), x.size(-1)), 
                            dim=[-3, -2, -1])
        
        return x


class FourierLayer(nn.Module):
    """
    Single Fourier Layer with skip connection
    """
    
    def __init__(self, channels: int, modes1: int, modes2: int, modes3: int):
        super().__init__()
        self.conv = SpectralConv3d(channels, channels, modes1, modes2, modes3)
        self.w = nn.Conv3d(channels, channels, 1)  # Point-wise convolution
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv(x)
        x2 = self.w(x)
        return F.gelu(x1 + x2)


class SkipBlock(nn.Module):
    """
    Dense Skip Connection Block (from Hainaut & Casas)
    Parallel MLP branch to capture high-frequency features
    """
    
    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.fc3 = nn.Linear(hidden_features, out_features)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [batch, in_features] - original input
        Returns:
            [batch, out_features] - skip connection output
        """
        x = F.gelu(self.fc1(z))
        x = F.gelu(self.fc2(x))
        x = self.fc3(x)
        return x


class PINO_Heston(nn.Module):
    """
    Physics-Informed Neural Operator for Heston Model
    Architecture: FNO backbone + ResNet-style skip connections
    """
    
    def __init__(self,
                 modes1: int = 12,
                 modes2: int = 12, 
                 modes3: int = 12,
                 width: int = 32,
                 n_layers: int = 4,
                 param_dim: int = 6,
                 coord_dim: int = 3,
                 skip_hidden: int = 128):
        """
        Args:
            modes1, modes2, modes3: Number of Fourier modes in each dimension
            width: Number of channels in Fourier layers
            n_layers: Number of Fourier layers
            param_dim: Dimension of Heston parameters (kappa, theta, sigma, rho, r, T)
            coord_dim: Dimension of coordinates (S, V, t)
            skip_hidden: Hidden dimension for skip connection blocks
        """
        super().__init__()
        
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.n_layers = n_layers
        self.param_dim = param_dim
        self.coord_dim = coord_dim
        
        # Input lifting: Coordinates + Parameters -> width channels
        self.lift = nn.Linear(coord_dim + param_dim, width)
        
        # Fourier layers
        self.fourier_layers = nn.ModuleList([
            FourierLayer(width, modes1, modes2, modes3) 
            for _ in range(n_layers)
        ])
        
        # Skip connection blocks (parallel ResNet branch)
        self.skip_blocks = nn.ModuleList([
            SkipBlock(coord_dim + param_dim, skip_hidden, width)
            for _ in range(n_layers)
        ])
        
        # Output projection: width channels -> 1 (option price)
        self.project = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, 1)
        )
        
    def coordinate_lift(self, coords: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Lift scalar parameters to vector fields (constant across grid)
        
        Args:
            coords: [batch, n_s, n_v, n_t, 3] - normalized coordinates
            params: [batch, 6] - Heston parameters
            
        Returns:
            [batch, n_s, n_v, n_t, 3+6] - concatenated input
        """
        batch, n_s, n_v, n_t, _ = coords.shape
        
        # Expand parameters to match spatial dimensions
        params_expanded = params.view(batch, 1, 1, 1, -1).expand(batch, n_s, n_v, n_t, -1)
        
        # Concatenate coordinates and parameters
        return torch.cat([coords, params_expanded], dim=-1)
    
    def forward(self, coords: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            coords: [batch, n_s, n_v, n_t, 3] - normalized coordinates (S, V, t)
            params: [batch, 6] - normalized Heston parameters
            
        Returns:
            [batch, n_s, n_v, n_t] - option prices
        """
        batch, n_s, n_v, n_t, _ = coords.shape
        
        # Coordinate lifting
        z = self.coordinate_lift(coords, params)  # [batch, n_s, n_v, n_t, 9]
        
        # Flatten spatial dimensions for input to lifting layer
        z_flat = z.view(-1, self.coord_dim + self.param_dim)  # [batch*n_s*n_v*n_t, 9]
        
        # Initial lifting
        x = self.lift(z_flat)  # [batch*n_s*n_v*n_t, width]
        x = x.view(batch, n_s, n_v, n_t, self.width)  # [batch, n_s, n_v, n_t, width]
        
        # Permute for conv3d: [batch, channels, n_s, n_v, n_t]
        x = x.permute(0, 4, 1, 2, 3)
        
        # Apply Fourier layers with skip connections
        for i, (fourier_layer, skip_block) in enumerate(zip(self.fourier_layers, self.skip_blocks)):
            # Fourier branch
            x_fourier = fourier_layer(x)
            
            # Skip branch (operates on original input z)
            skip_out = skip_block(z_flat)  # [batch*n_s*n_v*n_t, width]
            skip_out = skip_out.view(batch, n_s, n_v, n_t, self.width)
            skip_out = skip_out.permute(0, 4, 1, 2, 3)  # [batch, width, n_s, n_v, n_t]
            
            # Combine branches (as in H&C paper)
            x = x_fourier + skip_out
        
        # Permute back: [batch, n_s, n_v, n_t, width]
        x = x.permute(0, 2, 3, 4, 1)
        
        # Flatten for final projection
        x_flat = x.reshape(-1, self.width)
        
        # Project to option price
        out = self.project(x_flat)  # [batch*n_s*n_v*n_t, 1]
        out = out.view(batch, n_s, n_v, n_t)
        
        return out


if __name__ == "__main__":
    # Test the model
    print("Testing PINO_Heston model...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create dummy input
    batch_size = 2
    n_s, n_v, n_t = 32, 16, 16
    
    coords = torch.randn(batch_size, n_s, n_v, n_t, 3).to(device)
    params = torch.randn(batch_size, 6).to(device)
    
    # Initialize model
    model = PINO_Heston(
        modes1=8, modes2=8, modes3=8,
        width=32,
        n_layers=3
    ).to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {n_params:,}")
    
    # Forward pass
    with torch.no_grad():
        output = model(coords, params)
    
    print(f"\nInput shapes:")
    print(f"  Coords: {coords.shape}")
    print(f"  Params: {params.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    print("\n✓ Model test passed!")

"""
FIXED Model Architecture with Ansatz Transform and Sobolev Training
Addresses: Assumption 2 (Gibbs Phenomenon), Assumption 4 (Greeks Stability)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class AnsatzTransform(nn.Module):
    """
    Separates payoff kink from smooth time value
    
    Instead of learning V(S,t) directly, learn:
    V(S,t) = Payoff(S) + TimeValue(S,t)
    
    Where TimeValue is smooth and spectrally friendly
    """
    
    def __init__(self, strike: float = 100.0, option_type: str = 'call'):
        super().__init__()
        self.strike = strike
        self.option_type = option_type
    
    def compute_payoff(self, S: torch.Tensor) -> torch.Tensor:
        """Compute terminal payoff (handles kink)"""
        if self.option_type == 'call':
            return torch.maximum(S - self.strike, torch.zeros_like(S))
        else:
            return torch.maximum(self.strike - S, torch.zeros_like(S))
    
    def forward(self, S: torch.Tensor, time_value: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Combine payoff and time value
        
        Args:
            S: Spot prices [batch, n_s, n_v, n_t]
            time_value: Smooth component from PINO [batch, n_s, n_v, n_t]
            t: Time grid [batch, n_s, n_v, n_t]
            
        Returns:
            Total option value [batch, n_s, n_v, n_t]
        """
        payoff = self.compute_payoff(S)
        
        # At maturity (t=T), return pure payoff
        # Before maturity, add time value
        # Smooth interpolation using t
        total_value = payoff + time_value * (1 - t)
        
        return total_value


class SobolevLoss(nn.Module):
    """
    Sobolev training: regularize not just prices but also Greeks
    
    L = ||V - V_true||² + λ₁||∂V/∂S - Δ_true||² + λ₂||∂²V/∂S² - Γ_true||²
    
    Forces network to learn smooth pricing surface with accurate derivatives
    """
    
    def __init__(self,
                 data_generator,
                 lambda_price: float = 1.0,
                 lambda_delta: float = 0.1,
                 lambda_gamma: float = 0.01,
                 lambda_pde: float = 1.0,
                 lambda_bc: float = 1.0):
        super().__init__()
        
        from physics import HestonPDEResidual
        self.pde_computer = HestonPDEResidual(data_generator)
        self.data_generator = data_generator
        
        self.lambda_price = lambda_price
        self.lambda_delta = lambda_delta
        self.lambda_gamma = lambda_gamma
        self.lambda_pde = lambda_pde
        self.lambda_bc = lambda_bc
    
    def compute_greeks(self, P: torch.Tensor, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Delta and Gamma via automatic differentiation
        
        Args:
            P: Option prices [batch, n_s, n_v, n_t]
            coords: Normalized coordinates [batch, n_s, n_v, n_t, 3]
            
        Returns:
            delta: ∂P/∂S [batch, n_s, n_v, n_t]
            gamma: ∂²P/∂S² [batch, n_s, n_v, n_t]
        """
        # Extract S coordinate and denormalize
        S_norm = coords[..., 0]
        
        # Compute first derivative
        grad_outputs = torch.ones_like(P)
        delta_norm = torch.autograd.grad(
            P, coords,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )[0][..., 0]
        
        # Apply chain rule for normalization
        _, S_std = self.data_generator.normalization['S']
        delta = delta_norm / S_std
        
        # Compute second derivative
        gamma_norm = torch.autograd.grad(
            delta_norm, coords,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )[0][..., 0]
        
        gamma = gamma_norm / (S_std ** 2)
        
        return delta, gamma
    
    def forward(self,
                P: torch.Tensor,
                coords: torch.Tensor,
                params: torch.Tensor,
                boundary_data: dict,
                delta_target: Optional[torch.Tensor] = None,
                gamma_target: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        """
        Compute Sobolev loss
        
        Args:
            P: Predicted prices
            coords: Normalized coordinates
            params: Heston parameters
            boundary_data: Terminal boundary
            delta_target: Target Delta (optional, for supervised)
            gamma_target: Target Gamma (optional, for supervised)
            
        Returns:
            total_loss, loss_dict
        """
        # 1. PDE Residual Loss
        pde_residual = self.pde_computer.compute_residual(P, coords, params)
        loss_pde = torch.mean(pde_residual ** 2)
        
        # 2. Boundary Condition Loss
        P_terminal = P[:, :, :, -1]
        target_terminal = boundary_data['values']
        loss_bc = torch.mean((P_terminal - target_terminal) ** 2)
        
        # 3. Compute predicted Greeks
        delta_pred, gamma_pred = self.compute_greeks(P, coords)
        
        # 4. Greek Regularization
        # Even without targets, penalize large variations (smoothness)
        loss_delta_smooth = torch.mean((delta_pred[:, 1:, :, :] - delta_pred[:, :-1, :, :]) ** 2)
        loss_gamma_smooth = torch.mean((gamma_pred[:, 1:, :, :] - gamma_pred[:, :-1, :, :]) ** 2)
        
        # If targets provided, add supervised loss
        loss_delta_supervised = 0.0
        loss_gamma_supervised = 0.0
        
        if delta_target is not None:
            loss_delta_supervised = torch.mean((delta_pred - delta_target) ** 2)
        
        if gamma_target is not None:
            loss_gamma_supervised = torch.mean((gamma_pred - gamma_target) ** 2)
        
        # Total Greek loss
        loss_delta = loss_delta_smooth + loss_delta_supervised
        loss_gamma = loss_gamma_smooth + loss_gamma_supervised
        
        # 5. Total Sobolev Loss
        total_loss = (
            self.lambda_pde * loss_pde +
            self.lambda_bc * loss_bc +
            self.lambda_delta * loss_delta +
            self.lambda_gamma * loss_gamma
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'pde': loss_pde.item(),
            'boundary': loss_bc.item(),
            'delta': loss_delta.item(),
            'gamma': loss_gamma.item(),
            'delta_range': f"[{delta_pred.min():.4f}, {delta_pred.max():.4f}]",
            'gamma_range': f"[{gamma_pred.min():.6f}, {gamma_pred.max():.6f}]"
        }
        
        return total_loss, loss_dict


# UPDATE model_fixed.py

class PINO_Heston_Fixed(nn.Module):
    def __init__(self,
                 data_generator,  # <--- PASS GENERATOR HERE
                 modes1: int = 12,
                 modes2: int = 12,
                 modes3: int = 12,
                 width: int = 32,
                 n_layers: int = 4,
                 skip_hidden: int = 128,
                 use_ansatz: bool = True,
                 strike: float = 100.0,
                 option_type: str = 'call'):
        super().__init__()
        
        # Store normalization stats
        self.normalization = data_generator.normalization
        
        # Import base PINO architecture
        from model import PINO_Heston
        
        self.pino_base = PINO_Heston(
            modes1=modes1, 
            modes2=modes2, 
            modes3=modes3, 
            width=width, 
            n_layers=n_layers,
            skip_hidden=skip_hidden,
            param_dim=6,
            coord_dim=3
        )

        self.use_ansatz = use_ansatz
        if use_ansatz:
            self.ansatz = AnsatzTransform(strike, option_type)
            
        self.variance_transform = nn.Softplus()

    def forward(self, coords: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        # ... (Previous code for splitting coords) ...
        S_coord = coords[..., 0]
        V_coord = coords[..., 1]
        t_coord = coords[..., 2]

        # Apply softplus to Variance coordinate
        V_positive = self.variance_transform(V_coord)
        coords_safe = torch.stack([S_coord, V_positive, t_coord], dim=-1)

        # Get time value
        time_value = self.pino_base(coords_safe, params)

        if self.use_ansatz:
            # FIX: Use stored normalization stats instead of hardcoded 25/100
            S_mean, S_std = self.normalization['S']
            t_mean, t_std = self.normalization['t']
            
            # Denormalize S correctly
            S_denorm = coords[..., 0] * S_std + S_mean
            
            # Denormalize t correctly (map -1..1 back to 0..1 usually)
            # Assuming generator maps [0,1] -> [-1,1] via (x-0.5)/0.5
            t_denorm = coords[..., 2] * t_std + t_mean
            
            total_price = self.ansatz(S_denorm, time_value, t_denorm)
        else:
            total_price = time_value

        return total_price



if __name__ == "__main__":
    print("Testing Fixed PINO Architecture...")
    
    from data_generator_fixed import ImprovedHestonDataGenerator
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create generator
    generator = ImprovedHestonDataGenerator(n_s=32, n_v=16, n_t=16)
    
    # Sample parameters
    params_dict = generator.sample_params(batch_size=2)
    coords, params = generator.create_grid(params_dict, device)
    coords.requires_grad_(True)
    
    # Create model
    model = PINO_Heston_Fixed(
        modes1=8, modes2=8, modes3=8,
        width=16,
        n_layers=2,
        use_ansatz=True,
        strike=100.0
    ).to(device)
    
    # Forward pass
    prices = model(coords, params)
    
    print(f"\n1. Model Output:")
    print(f"   Prices shape: {prices.shape}")
    print(f"   Price range: [{prices.min():.4f}, {prices.max():.4f}]")
    print(f"   All non-negative: {(prices >= 0).all().item()}")
    
    # Test Sobolev loss
    print(f"\n2. Testing Sobolev Loss:")
    boundary = generator.generate_boundary_data(params_dict, strike=100.0, device=device)
    
    sobolev_loss = SobolevLoss(generator)
    total_loss, loss_dict = sobolev_loss(prices, coords, params, boundary)
    
    print(f"   Loss components:")
    for key, val in loss_dict.items():
        print(f"     {key}: {val}")
    
    print("\n✓ Fixed PINO architecture tests passed!")

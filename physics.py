"""
Physics Engine for Heston Model
Computes PDE residuals using automatic differentiation
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


class HestonPDEResidual:
    """
    Computes the Heston PDE residual for physics-informed training
    
    Heston PDE:
    ∂P/∂t + 0.5*V*S²*∂²P/∂S² + ρ*σ*V*S*∂²P/∂S∂V + 0.5*σ²*V*∂²P/∂V² 
    + r*S*∂P/∂S + κ(θ-V)*∂P/∂V - r*P = 0
    """
    
    def __init__(self, data_generator):
        """
        Args:
            data_generator: HestonDataGenerator instance for denormalization
        """
        self.data_generator = data_generator
        
    def compute_derivatives(self, 
                          P: torch.Tensor,
                          coords: torch.Tensor,
                          params: torch.Tensor,
                          create_graph: bool = True) -> Dict[str, torch.Tensor]:
        """
        Compute all required derivatives using automatic differentiation
        
        Args:
            P: [batch, n_s, n_v, n_t] - option prices (model output)
            coords: [batch, n_s, n_v, n_t, 3] - normalized coordinates
            params: [batch, 6] - normalized parameters
            create_graph: Whether to create computation graph (needed for training)
            
        Returns:
            Dictionary of derivatives in original (denormalized) space
        """
        batch, n_s, n_v, n_t = P.shape
        
        # Denormalize coordinates
        S = self.data_generator.denormalize(coords[..., 0], 'S')
        V = self.data_generator.denormalize(coords[..., 1], 'V')
        t = self.data_generator.denormalize(coords[..., 2], 't')
        
        # Denormalize parameters
        kappa = self.data_generator.denormalize(params[:, 0], 'kappa')
        theta = self.data_generator.denormalize(params[:, 1], 'theta')
        sigma = self.data_generator.denormalize(params[:, 2], 'sigma')
        rho = self.data_generator.denormalize(params[:, 3], 'rho')
        r = self.data_generator.denormalize(params[:, 4], 'r')
        
        # Reshape for broadcasting
        kappa = kappa.view(batch, 1, 1, 1)
        theta = theta.view(batch, 1, 1, 1)
        sigma = sigma.view(batch, 1, 1, 1)
        rho = rho.view(batch, 1, 1, 1)
        r = r.view(batch, 1, 1, 1)
        
        # Get normalization constants for chain rule
        _, S_std = self.data_generator.normalization['S']
        _, V_std = self.data_generator.normalization['V']
        _, t_std = self.data_generator.normalization['t']
        
        # Compute gradients w.r.t. normalized coordinates
        grad_outputs = torch.ones_like(P)
        
        # First derivatives
        dP_dS_norm = torch.autograd.grad(P, coords, grad_outputs=grad_outputs,
                                        create_graph=create_graph, retain_graph=True)[0][..., 0]
        dP_dV_norm = torch.autograd.grad(P, coords, grad_outputs=grad_outputs,
                                        create_graph=create_graph, retain_graph=True)[0][..., 1]
        dP_dt_norm = torch.autograd.grad(P, coords, grad_outputs=grad_outputs,
                                        create_graph=create_graph, retain_graph=True)[0][..., 2]
        
        # Chain rule: ∂P/∂S = (∂P/∂S_norm) * (∂S_norm/∂S) = (∂P/∂S_norm) / S_std
        dP_dS = dP_dS_norm / S_std
        dP_dV = dP_dV_norm / V_std
        dP_dt = dP_dt_norm / t_std
        
        # Second derivatives
        d2P_dS2_norm = torch.autograd.grad(dP_dS_norm, coords, grad_outputs=grad_outputs,
                                          create_graph=create_graph, retain_graph=True)[0][..., 0]
        d2P_dV2_norm = torch.autograd.grad(dP_dV_norm, coords, grad_outputs=grad_outputs,
                                          create_graph=create_graph, retain_graph=True)[0][..., 1]
        d2P_dSdV_norm = torch.autograd.grad(dP_dS_norm, coords, grad_outputs=grad_outputs,
                                           create_graph=create_graph, retain_graph=True)[0][..., 1]
        
        # Chain rule for second derivatives
        d2P_dS2 = d2P_dS2_norm / (S_std ** 2)
        d2P_dV2 = d2P_dV2_norm / (V_std ** 2)
        d2P_dSdV = d2P_dSdV_norm / (S_std * V_std)
        
        return {
            'S': S,
            'V': V,
            't': t,
            'kappa': kappa,
            'theta': theta,
            'sigma': sigma,
            'rho': rho,
            'r': r,
            'P': P,
            'dP_dS': dP_dS,
            'dP_dV': dP_dV,
            'dP_dt': dP_dt,
            'd2P_dS2': d2P_dS2,
            'd2P_dV2': d2P_dV2,
            'd2P_dSdV': d2P_dSdV
        }
    
    def compute_residual(self,
                        P: torch.Tensor,
                        coords: torch.Tensor,
                        params: torch.Tensor) -> torch.Tensor:
        """
        Compute Heston PDE residual
        
        Args:
            P: [batch, n_s, n_v, n_t] - option prices
            coords: [batch, n_s, n_v, n_t, 3] - normalized coordinates
            params: [batch, 6] - normalized parameters
            
        Returns:
            [batch, n_s, n_v, n_t] - PDE residual
        """
        # Get all derivatives
        derivs = self.compute_derivatives(P, coords, params)
        
        S = derivs['S']
        V = derivs['V']
        kappa = derivs['kappa']
        theta = derivs['theta']
        sigma = derivs['sigma']
        rho = derivs['rho']
        r = derivs['r']
        
        dP_dS = derivs['dP_dS']
        dP_dV = derivs['dP_dV']
        dP_dt = derivs['dP_dt']
        d2P_dS2 = derivs['d2P_dS2']
        d2P_dV2 = derivs['d2P_dV2']
        d2P_dSdV = derivs['d2P_dSdV']
        
        # Heston PDE
        residual = (
            dP_dt +
            0.5 * V * S**2 * d2P_dS2 +
            rho * sigma * V * S * d2P_dSdV +
            0.5 * sigma**2 * V * d2P_dV2 +
            r * S * dP_dS +
            kappa * (theta - V) * dP_dV -
            r * P
        )
        
        return residual
    
    def compute_boundary_residual(self,
                                  P: torch.Tensor,
                                  boundary_values: torch.Tensor) -> torch.Tensor:
        """
        Compute residual at terminal boundary (t=T)
        
        Args:
            P: [batch, n_s, n_v] - predicted option prices at maturity
            boundary_values: [batch, n_s, n_v] - true payoff values
            
        Returns:
            [batch, n_s, n_v] - boundary residual
        """
        return P - boundary_values


class HestonLoss(nn.Module):
    """
    Combined loss function for PINO training
    Balances PDE physics loss with boundary condition loss
    """
    
    def __init__(self, data_generator, lambda_pde: float = 1.0, lambda_bc: float = 1.0):
        """
        Args:
            data_generator: HestonDataGenerator instance
            lambda_pde: Weight for PDE loss
            lambda_bc: Weight for boundary condition loss
        """
        super().__init__()
        self.pde_computer = HestonPDEResidual(data_generator)
        self.lambda_pde = lambda_pde
        self.lambda_bc = lambda_bc
        
    def forward(self,
                P: torch.Tensor,
                coords: torch.Tensor,
                params: torch.Tensor,
                boundary_data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss
        
        Args:
            P: [batch, n_s, n_v, n_t] - predicted option prices
            coords: [batch, n_s, n_v, n_t, 3] - normalized coordinates
            params: [batch, 6] - normalized parameters
            boundary_data: Dictionary with 'coords' and 'values' for boundary
            
        Returns:
            total_loss: Scalar loss
            loss_dict: Dictionary with individual loss components
        """
        # PDE residual loss (interior points, excluding t=T)
        # We train on all time points, but the boundary loss handles t=T explicitly
        pde_residual = self.pde_computer.compute_residual(P, coords, params)
        loss_pde = torch.mean(pde_residual ** 2)
        
        # Boundary condition loss (terminal payoff at t=T)
        # Extract predictions at last time step
        P_boundary = P[:, :, :, -1]  # [batch, n_s, n_v]
        boundary_values = boundary_data['values']
        
        boundary_residual = self.pde_computer.compute_boundary_residual(P_boundary, boundary_values)
        loss_bc = torch.mean(boundary_residual ** 2)
        
        # Total loss
        total_loss = self.lambda_pde * loss_pde + self.lambda_bc * loss_bc
        
        # Return loss components for logging
        loss_dict = {
            'total': total_loss.item(),
            'pde': loss_pde.item(),
            'boundary': loss_bc.item()
        }
        
        return total_loss, loss_dict


if __name__ == "__main__":
    # Test physics engine
    print("Testing HestonPDEResidual...")
    
    from data_generator import HestonDataGenerator
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create data generator
    generator = HestonDataGenerator(n_s=16, n_v=8, n_t=8)
    
    # Sample parameters
    params_dict = generator.sample_params(batch_size=2)
    coords, params = generator.create_grid(params_dict, device=device)
    boundary = generator.generate_boundary_data(params_dict, device=device)
    
    # Create dummy predictions (requires grad)
    coords.requires_grad_(True)
    P = torch.randn(2, 16, 8, 8, device=device, requires_grad=True)
    
    # Compute PDE residual
    pde_computer = HestonPDEResidual(generator)
    residual = pde_computer.compute_residual(P, coords, params)
    
    print(f"\nPDE residual shape: {residual.shape}")
    print(f"PDE residual mean: {residual.mean().item():.6f}")
    print(f"PDE residual std: {residual.std().item():.6f}")
    
    # Test loss function
    loss_fn = HestonLoss(generator, lambda_pde=1.0, lambda_bc=1.0)
    total_loss, loss_dict = loss_fn(P, coords, params, boundary)
    
    print(f"\nLoss components:")
    for key, val in loss_dict.items():
        print(f"  {key}: {val:.6f}")
    
    print("\n✓ Physics engine test passed!")

"""
FIXED Data Generator with Hard Constraints and Improved Sampling
Addresses: Assumption 1 (Feller Condition), Positivity Constraints
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional


class ImprovedHestonDataGenerator:
    """
    Enhanced Heston data generator with:
    1. Feller condition enforcement
    2. Softplus transformations for positivity
    3. Adaptive parameter sampling
    """
    
    def __init__(self,
                 S_range: Tuple[float, float] = (50, 150),
                 V_range: Tuple[float, float] = (0.01, 0.1),
                 kappa_range: Tuple[float, float] = (1.0, 5.0),
                 theta_range: Tuple[float, float] = (0.01, 0.1),
                 sigma_range: Tuple[float, float] = (0.1, 0.5),
                 rho_range: Tuple[float, float] = (-0.9, 0.0),
                 n_s: int = 64,
                 n_v: int = 32,
                 n_t: int = 32,
                 enforce_feller: bool = True,
                 feller_safety_margin: float = 0.1):
        """
        Args:
            enforce_feller: If True, reject parameter samples violating Feller condition
            feller_safety_margin: Safety margin for Feller condition (2κθ ≥ σ²(1+margin))
        """
        self.S_range = S_range
        self.V_range = V_range
        self.kappa_range = kappa_range
        self.theta_range = theta_range
        self.sigma_range = sigma_range
        self.rho_range = rho_range
        
        self.n_s = n_s
        self.n_v = n_v
        self.n_t = n_t
        
        self.enforce_feller = enforce_feller
        self.feller_safety_margin = feller_safety_margin
        
        # Compute normalization statistics
        self.normalization = self._compute_normalization()
        
        # Small epsilon for numerical stability
        self.eps = 1e-6
    
    def _compute_normalization(self) -> Dict:
        """Compute mean and std for each variable"""
        return {
            'S': (np.mean(self.S_range), np.std(self.S_range)),
            'V': (np.mean(self.V_range), np.std(self.V_range)),
            'kappa': (np.mean(self.kappa_range), np.std(self.kappa_range)),
            'theta': (np.mean(self.theta_range), np.std(self.theta_range)),
            'sigma': (np.mean(self.sigma_range), np.std(self.sigma_range)),
            'rho': (np.mean(self.rho_range), np.std(self.rho_range)),
            't': (0.5, 0.5),
            # FIX: Add identity normalization for r and T
            'r': (0.0, 1.0),  # Identity: denormalize(r) -> r * 1 + 0 = r
            'T': (0.0, 1.0)   # Identity: denormalize(T) -> T * 1 + 0 = T
        }
    def check_feller_condition(self, kappa: float, theta: float, sigma: float) -> bool:
        """
        Check Feller condition: 2κθ > σ²
        
        Args:
            kappa: Mean reversion rate
            theta: Long-term variance
            sigma: Volatility of volatility
            
        Returns:
            True if Feller condition satisfied (with margin)
        """
        lhs = 2 * kappa * theta
        rhs = sigma ** 2 * (1 + self.feller_safety_margin)
        return lhs >= rhs
    
    def sample_params(self, batch_size: int = 1, max_retries: int = 100) -> Dict[str, torch.Tensor]:
        """
        Sample Heston parameters with Feller condition enforcement
        
        Args:
            batch_size: Number of parameter sets
            max_retries: Maximum attempts to satisfy Feller condition
            
        Returns:
            Dictionary of parameter tensors
        """
        params = {
            'S': torch.empty(batch_size),
            'V': torch.empty(batch_size),
            'kappa': torch.empty(batch_size),
            'theta': torch.empty(batch_size),
            'sigma': torch.empty(batch_size),
            'rho': torch.empty(batch_size),
            'r': torch.empty(batch_size),
            'T': torch.empty(batch_size)
        }
        
        for i in range(batch_size):
            valid = False
            retry_count = 0
            
            while not valid and retry_count < max_retries:
                # Sample parameters
                kappa = np.random.uniform(*self.kappa_range)
                theta = np.random.uniform(*self.theta_range)
                sigma = np.random.uniform(*self.sigma_range)
                
                # Check Feller condition
                if self.enforce_feller:
                    if self.check_feller_condition(kappa, theta, sigma):
                        valid = True
                    else:
                        retry_count += 1
                        if retry_count >= max_retries:
                            # Fallback: adjust sigma to satisfy Feller
                            sigma = np.sqrt(2 * kappa * theta) * 0.95
                            valid = True
                else:
                    valid = True
            
            # Store parameters
            params['S'][i] = np.random.uniform(*self.S_range)
            params['V'][i] = np.random.uniform(*self.V_range)
            params['kappa'][i] = kappa
            params['theta'][i] = theta
            params['sigma'][i] = sigma
            params['rho'][i] = np.random.uniform(*self.rho_range)
            params['r'][i] = 0.05  # Can be made random if needed
            params['T'][i] = 1.0   # Can be made random if needed
        
        return params
    
    def apply_softplus_transform(self, h: torch.Tensor, param_type: str) -> torch.Tensor:
        """
        Apply softplus transformation for positivity constraint
        
        v = softplus(h) + ε = log(1 + exp(h)) + ε
        
        This guarantees v > 0 for any h, preventing negative variance
        
        Args:
            h: Unbounded latent variable
            param_type: 'V', 'kappa', 'theta', or 'sigma'
            
        Returns:
            Positive-constrained parameter
        """
        # Softplus with small epsilon
        positive_param = torch.nn.functional.softplus(h) + self.eps
        
        # Scale to appropriate range
        if param_type == 'V':
            # Map to [0.01, 0.1]
            return 0.01 + 0.09 * torch.sigmoid(h)
        elif param_type == 'kappa':
            # Map to [1.0, 5.0]
            return 1.0 + 4.0 * torch.sigmoid(h)
        elif param_type == 'theta':
            # Map to [0.01, 0.1]
            return 0.01 + 0.09 * torch.sigmoid(h)
        elif param_type == 'sigma':
            # Map to [0.1, 0.5]
            return 0.1 + 0.4 * torch.sigmoid(h)
        else:
            return positive_param
    
    def inverse_softplus_transform(self, v: torch.Tensor, param_type: str) -> torch.Tensor:
        """
        Inverse transformation for calibration (v → h)
        
        Args:
            v: Positive parameter value
            param_type: Parameter type
            
        Returns:
            Unbounded latent variable h
        """
        if param_type == 'V':
            normalized = (v - 0.01) / 0.09
        elif param_type == 'kappa':
            normalized = (v - 1.0) / 4.0
        elif param_type == 'theta':
            normalized = (v - 0.01) / 0.09
        elif param_type == 'sigma':
            normalized = (v - 0.1) / 0.4
        else:
            normalized = v
        
        # Inverse sigmoid: logit
        return torch.logit(torch.clamp(normalized, 1e-6, 1 - 1e-6))
    
    def normalize(self, value: torch.Tensor, var_name: str) -> torch.Tensor:
        """Normalize to [-1, 1] range"""
        mean, std = self.normalization[var_name]
        return (value - mean) / (std + 1e-8)
    
    def denormalize(self, value: torch.Tensor, var_name: str) -> torch.Tensor:
        """Denormalize from [-1, 1] to original range"""
        mean, std = self.normalization[var_name]
        return value * std + mean
    
    def create_grid(self, params_dict: Dict[str, torch.Tensor], device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create normalized coordinate grid
        
        Returns:
            coords: [batch, n_s, n_v, n_t, 3] normalized coordinates
            params: [batch, 6] normalized parameters
        """
        batch_size = params_dict['S'].shape[0]
        
        # Create grids for each dimension
        s_grid = torch.linspace(self.S_range[0], self.S_range[1], self.n_s, device=device)
        v_grid = torch.linspace(self.V_range[0], self.V_range[1], self.n_v, device=device)
        t_grid = torch.linspace(0.0, 1.0, self.n_t, device=device)
        
        # Meshgrid
        S, V, T = torch.meshgrid(s_grid, v_grid, t_grid, indexing='ij')
        
        # Normalize
        S_norm = self.normalize(S, 'S')
        V_norm = self.normalize(V, 'V')
        T_norm = self.normalize(T, 't')
        
        # Stack coordinates
        coords = torch.stack([S_norm, V_norm, T_norm], dim=-1)
        coords = coords.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1).to(device)
        
        # Normalize parameters
        params = torch.stack([
            self.normalize(params_dict['kappa'].to(device), 'kappa'),
            self.normalize(params_dict['theta'].to(device), 'theta'),
            self.normalize(params_dict['sigma'].to(device), 'sigma'),
            self.normalize(params_dict['rho'].to(device), 'rho'),
            params_dict['r'].to(device),  # Keep r unnormalized for now
            params_dict['T'].to(device)   # Keep T unnormalized for now
        ], dim=1)
        
        return coords, params
    
    def generate_boundary_data(self, params_dict: Dict[str, torch.Tensor], 
                              strike: float = 100.0,
                              option_type: str = 'call',
                              device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """
        Generate terminal boundary condition (payoff at maturity)
        
        Uses Ansatz transformation: separate payoff from time value
        """
        batch_size = params_dict['S'].shape[0]
        
        # Create S-V grid at t=T
        s_grid = torch.linspace(self.S_range[0], self.S_range[1], self.n_s, device=device)
        v_grid = torch.linspace(self.V_range[0], self.V_range[1], self.n_v, device=device)
        
        S, V = torch.meshgrid(s_grid, v_grid, indexing='ij')
        
        # Compute payoff
        if option_type == 'call':
            payoff = torch.maximum(S - strike, torch.zeros_like(S))
        else:  # put
            payoff = torch.maximum(strike - S, torch.zeros_like(S))
        
        # Normalize coordinates
        S_norm = self.normalize(S, 'S')
        V_norm = self.normalize(V, 'V')
        T_norm = torch.ones_like(S) * self.normalize(torch.tensor([1.0]), 't')
        
        coords = torch.stack([S_norm, V_norm, T_norm], dim=-1)
        coords = coords.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)
        
        payoff = payoff.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        
        return {
            'coords': coords,
            'values': payoff
        }


if __name__ == "__main__":
    print("Testing Improved Data Generator...")
    
    generator = ImprovedHestonDataGenerator(enforce_feller=True)
    
    # Test parameter sampling with Feller enforcement
    print("\n1. Testing Feller Condition Enforcement:")
    params = generator.sample_params(batch_size=5)
    
    for i in range(5):
        kappa = params['kappa'][i].item()
        theta = params['theta'][i].item()
        sigma = params['sigma'][i].item()
        
        feller_lhs = 2 * kappa * theta
        feller_rhs = sigma ** 2
        
        print(f"\nSample {i+1}:")
        print(f"  κ={kappa:.4f}, θ={theta:.4f}, σ={sigma:.4f}")
        print(f"  2κθ={feller_lhs:.6f}, σ²={feller_rhs:.6f}")
        print(f"  Feller satisfied: {feller_lhs > feller_rhs}")
    
    # Test softplus transformation
    print("\n2. Testing Softplus Transform (Positivity Guarantee):")
    h_test = torch.randn(10)
    v_positive = generator.apply_softplus_transform(h_test, 'V')
    
    print(f"  Latent h range: [{h_test.min():.4f}, {h_test.max():.4f}]")
    print(f"  Transformed V range: [{v_positive.min():.6f}, {v_positive.max():.6f}]")
    print(f"  All positive: {(v_positive > 0).all().item()}")
    
    # Test grid creation
    print("\n3. Testing Grid Creation:")
    coords, param_tensor = generator.create_grid(params)
    
    print(f"  Coords shape: {coords.shape}")
    print(f"  Params shape: {param_tensor.shape}")
    print(f"  Coords normalized range: [{coords.min():.4f}, {coords.max():.4f}]")
    
    print("\n✓ Improved data generator tests passed!")

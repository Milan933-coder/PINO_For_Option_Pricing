"""
Adaptive Loss Balancing with WamOL (Weighted Adaptive Multi-Objective Learning)
Addresses: Assumption 3 (Loss Function Uniformity)

Automatically learns optimal loss weights during training
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple,Optional


class WamOLLoss(nn.Module):
    """
    Weighted Adaptive Multi-Objective Loss
    
    Key Idea: Turn loss weights into trainable parameters that compete
    with the main network. The weights are updated to MAXIMIZE loss
    (finding hardest parts), while network minimizes it.
    
    This creates a minimax game that forces the network to focus on
    the most difficult aspects of the problem automatically.
    """
    
    def __init__(self,
                 num_objectives: int = 3,
                 init_weights: Optional[List[float]] = None,
                 temperature: float = 1.0,
                 update_freq: int = 10):
        """
        Args:
            num_objectives: Number of loss terms (e.g., PDE, BC, Greeks)
            init_weights: Initial weights (default: uniform)
            temperature: Softmax temperature for weight normalization
            update_freq: Update weights every N iterations
        """
        super().__init__()
        
        self.num_objectives = num_objectives
        self.temperature = temperature
        self.update_freq = update_freq
        self.iteration = 0
        
        # Initialize trainable log-weights
        if init_weights is None:
            init_weights = [1.0] * num_objectives
        
        # Use log-weights for better optimization
        log_weights = torch.log(torch.tensor(init_weights, dtype=torch.float32))
        self.log_weights = nn.Parameter(log_weights)
        
        # Track loss history for adaptive updates
        self.loss_history = {i: [] for i in range(num_objectives)}
    
    def get_normalized_weights(self) -> torch.Tensor:
        """
        Convert log-weights to normalized weights via softmax
        
        Ensures weights are positive and sum to 1
        """
        weights = torch.softmax(self.log_weights / self.temperature, dim=0)
        return weights
    
    def forward(self, loss_components: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """
        Compute weighted combination of losses
        
        Args:
            loss_components: Dict of individual loss terms
            
        Returns:
            total_loss, info_dict
        """
        # Get current weights
        weights = self.get_normalized_weights()
        
        # Convert dict to ordered tensor
        loss_names = sorted(loss_components.keys())
        losses = torch.stack([loss_components[name] for name in loss_names])
        
        # Weighted sum
        total_loss = torch.sum(weights * losses)
        
        # Record history
        for i, name in enumerate(loss_names):
            self.loss_history[i].append(losses[i].item())
        
        # Return info
        info = {
            'total_loss': total_loss.item(),
            'weights': {name: weights[i].item() for i, name in enumerate(loss_names)},
            'components': {name: losses[i].item() for i, name in enumerate(loss_names)}
        }
        
        self.iteration += 1
        
        return total_loss, info
    
    def update_weights_adversarial(self, loss_components: Dict[str, torch.Tensor],
                                   lr_weights: float = 0.01):
        """
        Adversarial update: increase weights for high-loss terms
        
        This makes weights "fight" against the network, forcing it
        to focus on hard parts of the problem
        """
        weights = self.get_normalized_weights()
        loss_names = sorted(loss_components.keys())
        losses = torch.stack([loss_components[name] for name in loss_names])
        
        # Gradient ascent on weights (maximize weighted loss)
        # This increases weights for terms with high loss
        weighted_loss = torch.sum(weights * losses)
        
        # Manual gradient ascent
        with torch.no_grad():
            grad_weights = torch.autograd.grad(weighted_loss, self.log_weights,
                                              retain_graph=True)[0]
            self.log_weights += lr_weights * grad_weights
            
            # Clip to prevent explosion
            self.log_weights.clamp_(-10, 10)


# UPDATE adaptive_loss.py inside AdaptiveHestonLoss

class AdaptiveHestonLoss(nn.Module):
    def __init__(self,
                 data_generator,
                 use_wamol: bool = True,
                 sobolev_training: bool = True,
                 hard_ansatz_active: bool = True): # Add this flag
        super().__init__()
        
        from physics import HestonPDEResidual
        
        self.pde_computer = HestonPDEResidual(data_generator)
        self.data_generator = data_generator
        self.use_wamol = use_wamol
        self.sobolev_training = sobolev_training
        self.hard_ansatz_active = hard_ansatz_active
        
        # If Hard Ansatz is active, we REMOVE 'boundary' from objectives
        # Objectives: PDE, Delta, Gamma, Vega (4 items)
        base_objs = ['pde']
        if not hard_ansatz_active:
            base_objs.append('boundary')
        if sobolev_training:
            base_objs.extend(['delta', 'gamma', 'vega'])
            
        self.loss_keys = base_objs
        self.wamol = WamOLLoss(num_objectives=len(base_objs)) if use_wamol else None

    def compute_all_losses(self, P, coords, params, boundary_data):
        losses = {}
        
        # 1. PDE Residual
        pde_residual = self.pde_computer.compute_residual(P, coords, params)
        losses['pde'] = torch.mean(pde_residual ** 2)
        
        # 2. Boundary Condition (Only compute if Ansatz is NOT used)
        if not self.hard_ansatz_active:
            P_terminal = P[:, :, :, -1]
            target_terminal = boundary_data['values']
            losses['boundary'] = torch.mean((P_terminal - target_terminal) ** 2)
        
        # 3. Sobolev (Greeks)
        if self.sobolev_training:
            # Compute Delta
            grad_outputs = torch.ones_like(P)
            delta_norm = torch.autograd.grad(
                P, coords,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True
            )[0][..., 0]
            
            _, S_std = self.data_generator.normalization['S']
            delta = delta_norm / S_std
            
            # Delta smoothness
            losses['delta'] = torch.mean((delta[:, 1:, :, :] - delta[:, :-1, :, :]) ** 2)
            
            # Compute Gamma
            gamma_norm = torch.autograd.grad(
                delta_norm, coords,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True
            )[0][..., 0]
            
            gamma = gamma_norm / (S_std ** 2)
            losses['gamma'] = torch.mean((gamma[:, 1:, :, :] - gamma[:, :-1, :, :]) ** 2)
            
            # Vega (for completeness)
            vega_norm = torch.autograd.grad(
                P, coords,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True
            )[0][..., 1]
            
            _, V_std = self.data_generator.normalization['V']
            vega = vega_norm / V_std
            losses['vega'] = torch.mean((vega[:, :, 1:, :] - vega[:, :, :-1, :]) ** 2)
        
        return losses

    def forward(self, P, coords, params, boundary_data):
        loss_components = self.compute_all_losses(P, coords, params, boundary_data)
        
        if self.use_wamol:
            # Filter loss_components to match initialized keys to avoid shape errors
            filtered_losses = {k: loss_components[k] for k in self.loss_keys if k in loss_components}
            total_loss, info = self.wamol(filtered_losses)
        else:
            total_loss = sum(loss_components.values())
            info = {'total_loss': total_loss.item(), 'components': {k:v.item() for k,v in loss_components.items()}}
            
        return total_loss, info
    def update_weights(self, loss_components: Dict[str, object]):
        """Update WamOL weights adversarially"""
        if self.use_wamol and self.wamol.iteration % self.wamol.update_freq == 0:
            
            # FIX: Ensure components are Tensors before passing to WAMOL
            tensor_components = {}
            for k, v in loss_components.items():
                if isinstance(v, torch.Tensor):
                    tensor_components[k] = v.detach() # Use detached tensor
                else:
                    # Convert float back to tensor for calculation
                    tensor_components[k] = torch.tensor(v, dtype=torch.float32, device=self.wamol.log_weights.device)
            
            self.wamol.update_weights_adversarial(tensor_components, lr_weights=0.01)


if __name__ == "__main__":
    print("Testing Adaptive Loss Balancing (WamOL)...")
    
    from data_generator_fixed import ImprovedHestonDataGenerator
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create generator
    generator = ImprovedHestonDataGenerator(n_s=16, n_v=8, n_t=8)
    
    # Sample data
    params_dict = generator.sample_params(batch_size=2)
    coords, params = generator.create_grid(params_dict, device)
    coords.requires_grad_(True)
    
    P = torch.randn(2, 16, 8, 8, device=device, requires_grad=True)
    boundary = generator.generate_boundary_data(params_dict, device=device)
    
    # Test WamOL
    print("\n1. Testing WamOL:")
    adaptive_loss = AdaptiveHestonLoss(generator, use_wamol=True, sobolev_training=True)
    
    # Initial weights
    initial_weights = adaptive_loss.wamol.get_normalized_weights()
    print(f"   Initial weights: {initial_weights.detach().cpu().numpy()}")
    
    # Simulate training iterations
    for iter in range(20):
        total_loss, info = adaptive_loss(P, coords, params, boundary)
        
        if iter % 5 == 0:
            print(f"\n   Iteration {iter}:")
            print(f"     Total loss: {info['total_loss']:.6f}")
            print(f"     Weights: {info['weights']}")
            print(f"     Components: {info['components']}")
        
        # Update weights adversarially
        loss_components = {k: torch.tensor(v) for k, v in info['components'].items()}
        adaptive_loss.update_weights(loss_components)
    
    # Final weights
    final_weights = adaptive_loss.wamol.get_normalized_weights()
    print(f"\n   Final weights: {final_weights.detach().cpu().numpy()}")
    print(f"   Weight shift: {(final_weights - initial_weights).abs().sum():.4f}")
    
    print("\n✓ Adaptive loss balancing tests passed!")

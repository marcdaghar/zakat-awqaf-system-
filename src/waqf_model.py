"""
Awqaf Model - Islamic Endowment System.
Implements waqf dynamics, benefits, and social services provision.
"""

import numpy as np

class WaqfModel:
    """
    Awqaf (Islamic endowments) model.
    
    Key relationships:
        dW/dt = φ·S - ψ·W + Donations  # Waqf dynamics
        B(t) = ω·W(t) - χ·B(t)  # Benefits
        Services(t) = ∫B(τ)dτ  # Social services
    """
    
    def __init__(self, phi=0.05, psi=0.02, omega=0.08, chi=0.03,
                 S_base=100.0, donation_rate=0.01):
        """
        Args:
            phi: Waqf creation rate from surplus
            psi: Depreciation/maintenance rate
            omega: Return rate on waqf assets
            chi: Maintenance cost rate
            S_base: Base surplus for waqf creation
            donation_rate: Donation rate
        """
        self.phi = phi
        self.psi = psi
        self.omega = omega
        self.chi = chi
        self.S_base = S_base
        self.donation_rate = donation_rate
    
    def surplus(self, t):
        """Surplus function for waqf creation."""
        # Cyclical surplus with some noise
        return self.S_base * (1 + 0.3 * np.sin(2 * np.pi * t / 14))
    
    def donations(self, W, t):
        """Donation function."""
        return self.donation_rate * W
    
    def waqf_dynamics(self, W, t):
        """
        Waqf dynamics: dW/dt = φ·S - ψ·W + Donations
        """
        S = self.surplus(t)
        D = self.donations(W, t)
        dW = self.phi * S - self.psi * W + D
        return dW
    
    def benefits(self, W, B, t):
        """
        Benefits from waqf: B(t) = ω·W(t) - χ·B(t)
        """
        return self.omega * W - self.chi * B
    
    def simulate_waqf(self, W0=500.0, B0=50.0, n_years=50, dt=0.1,
                      verbose=True):
        """
        Simulate waqf system.
        
        Returns:
            Dictionary with results
        """
        n_steps = int(n_years / dt)
        
        # Initialize
        W = np.zeros(n_steps)
        B = np.zeros(n_steps)
        Services = np.zeros(n_steps)
        S = np.zeros(n_steps)
        D = np.zeros(n_steps)
        
        W[0] = W0
        B[0] = B0
        
        for step in range(1, n_steps):
            t = step * dt
            
            # Surplus
            S[step] = self.surplus(t)
            
            # Donations
            D[step] = self.donations(W[step-1], t)
            
            # Waqf dynamics
            dW = self.waqf_dynamics(W[step-1], t)
            W[step] = max(0, W[step-1] + dW * dt)
            
            # Benefits
            dB = self.benefits(W[step], B[step-1], t)
            B[step] = max(0, B[step-1] + dB * dt)
            
            # Cumulative services
            Services[step] = Services[step-1] + B[step] * dt
        
        return {
            'W': W,
            'B': B,
            'Services': Services,
            'S': S,
            'D': D,
            't': np.arange(0, n_years, dt)
        }
    
    def compute_metrics(self, results):
        """
        Compute metrics from waqf simulation.
        """
        W = results['W']
        B = results['B']
        Services = results['Services']
        
        return {
            'final_waqf': W[-1],
            'final_benefits': B[-1],
            'total_services': Services[-1],
            'average_benefits': np.mean(B),
            'max_benefits': np.max(B),
            'min_benefits': np.min(B),
            'waqf_growth': (W[-1] - W[0]) / W[0]
        }
    
    def compute_waqf_lifecycle(self, n_periods=100):
        """
        Compute the waqf lifecycle stages.
        """
        lifecycle = {
            'Creation': np.random.exponential(5, n_periods),
            'Asset Accumulation': np.random.normal(10, 2, n_periods),
            'Income Generation': np.random.normal(15, 3, n_periods),
            'Social Services': np.random.normal(20, 4, n_periods)
        }
        
        # Normalize
        for key in lifecycle:
            lifecycle[key] = lifecycle[key] / np.sum(lifecycle[key])
        
        return lifecycle

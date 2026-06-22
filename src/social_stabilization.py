"""
Social Stabilization Model.
Combines Zakat, Awqaf, and social safety nets.
"""

import numpy as np
from src.zakat_model import ZakatModel
from src.waqf_model import WaqfModel

class SocialStabilizationModel:
    """
    Combined social stabilization model with Zakat and Awqaf.
    """
    
    def __init__(self, zakat_model=None, waqf_model=None):
        self.zakat_model = zakat_model or ZakatModel()
        self.waqf_model = waqf_model or WaqfModel()
    
    def simulate_full_system(self, wealth_initial, n_years=50, dt=0.1,
                             with_zakat=True, with_waqf=True):
        """
        Simulate the full system with Zakat and Awqaf.
        
        Args:
            wealth_initial: Initial wealth distribution
            n_years: Number of years
            dt: Time step
            with_zakat: Include Zakat mechanism
            with_waqf: Include Awqaf mechanism
        
        Returns:
            Dictionary with results
        """
        n_steps = int(n_years / dt)
        n_agents = len(wealth_initial)
        
        # Initialize
        wealth = wealth_initial.copy()
        gini_history = np.zeros(n_steps)
        consumption_history = np.zeros(n_steps)
        social_services = np.zeros(n_steps)
        waqf_stock = np.zeros(n_steps)
        zakat_collected = np.zeros(n_steps)
        
        # Waqf variables
        W_waqf = 500.0
        B_waqf = 50.0
        
        # Store initial
        gini_history[0] = self.zakat_model.compute_gini(wealth)
        consumption_history[0] = np.mean(wealth) * 0.05
        waqf_stock[0] = W_waqf
        
        for step in range(1, n_steps):
            t = step * dt
            
            # --- Zakat ---
            if with_zakat:
                total_wealth = np.sum(wealth)
                Z = self.zakat_model.total_zakat(total_wealth)
                zakat_collected[step] = Z
                
                # Distribute to eligible (bottom 30%)
                dist = self.zakat_model.distribution_by_category(Z)
                sorted_indices = np.argsort(wealth)
                n_eligible = int(0.3 * n_agents)
                eligible_indices = sorted_indices[:n_eligible]
                
                redistribution = np.zeros(n_agents)
                if n_eligible > 0:
                    need = 1.0 / (wealth[eligible_indices] + 1)
                    need = need / np.sum(need)
                    redistribution[eligible_indices] = dist[0] * need + dist[1] * need / 2
                
                zakat_per_agent = self.zakat_model.tau * wealth
            else:
                zakat_per_agent = np.zeros(n_agents)
                redistribution = np.zeros(n_agents)
                zakat_collected[step] = 0
            
            # --- Awqaf ---
            if with_waqf:
                dW_waqf = self.waqf_model.waqf_dynamics(W_waqf, t)
                W_waqf = max(0, W_waqf + dW_waqf * dt)
                dB_waqf = self.waqf_model.benefits(W_waqf, B_waqf, t)
                B_waqf = max(0, B_waqf + dB_waqf * dt)
                social_services[step] = B_waqf
            else:
                W_waqf = 0
                B_waqf = 0
                social_services[step] = 0
            
            waqf_stock[step] = W_waqf
            
            # --- Wealth dynamics ---
            consumption_base = 10.0
            for i in range(n_agents):
                dW = (0.05 * wealth[i] - zakat_per_agent[i] - 
                      consumption_base + redistribution[i])
                wealth[i] = max(1, wealth[i] + dW * dt)
            
            # --- Metrics ---
            gini_history[step] = self.zakat_model.compute_gini(wealth)
            consumption_history[step] = np.mean(wealth) * 0.05
        
        return {
            'wealth_final': wealth,
            'gini_history': gini_history,
            'consumption_history': consumption_history,
            'social_services': social_services,
            'waqf_stock': waqf_stock,
            'zakat_collected': zakat_collected,
            't': np.arange(0, n_years, dt)
        }
    
    def compute_consumption_shock(self, baseline, shock_amplitude=0.3,
                                 shock_start=20, shock_duration=5):
        """
        Simulate a consumption shock.
        """
        n = len(baseline)
        shock = np.zeros(n)
        
        for i in range(n):
            if shock_start <= i * 0.1 <= shock_start + shock_duration:
                shock[i] = -shock_amplitude * baseline[i]
            else:
                shock[i] = 0
        
        return shock

"""
Zakat Model - Islamic Redistributive Mechanism.
Implements Zakat collection, distribution, and wealth dynamics.
"""

import numpy as np
from scipy.stats import lognorm, pareto

class ZakatModel:
    """
    Zakat model with wealth dynamics and redistribution.
    
    Key relationships:
        Z(t) = τ · W(t)  # Total Zakat
        R_i(t) = ρ_i · Z(t)  # Distribution to 8 categories
        dW_i/dt = r_i·W_i - Z_i - C_i + R_i  # Wealth dynamics
        G_post = (1-δ)·G_pre  # Gini reduction
    """
    
    def __init__(self, tau=0.025, rho=None, r_return=0.05, C_base=10.0):
        """
        Args:
            tau: Zakat rate (default 2.5%)
            rho: Distribution weights for 8 categories
            r_return: Rate of return on wealth
            C_base: Base consumption
        """
        self.tau = tau
        self.r_return = r_return
        self.C_base = C_base
        
        # Default distribution weights (approximate)
        if rho is None:
            self.rho = np.array([
                0.20,  # Poor (al-fuqara')
                0.20,  # Needy (al-masakin)
                0.05,  # Collectors ('amilin)
                0.10,  # Reconciliation (mu'allafat)
                0.05,  # Bondage (fi al-riqab)
                0.15,  # Debtors (al-gharimin)
                0.15,  # Cause of Allah (fi sabil Allah)
                0.10   # Wayfarer (ibn al-sabil)
            ])
        else:
            self.rho = np.array(rho)
        
        # Ensure sum = 1
        self.rho = self.rho / np.sum(self.rho)
        
        # Category names
        self.categories = [
            'Poor (al-fuqara\')',
            'Needy (al-masakin)',
            'Zakat Collectors',
            'Reconciliation',
            'Bondage',
            'Debtors',
            'Cause of Allah',
            'Wayfarer'
        ]
    
    def total_zakat(self, W):
        """Total Zakat collected: Z(t) = τ · W(t)"""
        return self.tau * W
    
    def distribution_by_category(self, Z):
        """Distribution by category: R_i(t) = ρ_i · Z(t)"""
        return self.rho * Z
    
    def wealth_dynamics(self, W, t, zakat_paid=0, zakat_received=0, consumption=None):
        """
        Wealth dynamics: dW/dt = r·W - Z - C + R
        
        Args:
            W: Current wealth
            t: Time
            zakat_paid: Zakat paid by this agent
            zakat_received: Zakat received by this agent
            consumption: Consumption (default: C_base)
        """
        if consumption is None:
            consumption = self.C_base
        
        dW = self.r_return * W - zakat_paid - consumption + zakat_received
        return dW
    
    def generate_wealth_distribution(self, n_agents=10000, method='lognormal'):
        """
        Generate initial wealth distribution.
        
        Args:
            n_agents: Number of agents
            method: 'lognormal', 'pareto', or 'uniform'
        
        Returns:
            Array of wealth values
        """
        if method == 'lognormal':
            # Log-normal distribution (typical for wealth)
            wealth = lognorm.rvs(s=1.5, scale=100, size=n_agents)
        elif method == 'pareto':
            # Pareto distribution (more unequal)
            wealth = pareto.rvs(b=2.0, scale=100, size=n_agents)
        else:
            # Uniform distribution
            wealth = np.random.uniform(1, 1000, n_agents)
        
        return np.clip(wealth, 1, None)
    
    def simulate_zakat(self, wealth_initial, n_years=50, dt=0.1, verbose=True):
        """
        Simulate Zakat dynamics over time.
        
        Args:
            wealth_initial: Initial wealth distribution (array)
            n_years: Number of years to simulate
            dt: Time step
        
        Returns:
            Dictionary with results
        """
        n_steps = int(n_years / dt)
        n_agents = len(wealth_initial)
        
        # Initialize
        wealth = wealth_initial.copy()
        zakat_collected = np.zeros(n_steps)
        zakat_distributed = np.zeros(n_steps)
        zakat_by_category = np.zeros((n_steps, 8))
        gini_history = np.zeros(n_steps)
        wealth_history = np.zeros((n_steps, n_agents))
        
        # Store initial state
        wealth_history[0] = wealth.copy()
        gini_history[0] = self.compute_gini(wealth)
        
        for step in range(1, n_steps):
            t = step * dt
            
            # Total wealth
            total_wealth = np.sum(wealth)
            
            # Collect Zakat
            Z = self.total_zakat(total_wealth)
            zakat_collected[step] = Z
            
            # Distribute Zakat by category
            dist = self.distribution_by_category(Z)
            zakat_distributed[step] = np.sum(dist)
            zakat_by_category[step] = dist
            
            # Calculate Zakat per agent (proportional to wealth)
            zakat_per_agent = self.tau * wealth
            
            # Distribute to eligible agents (poorest 30% receive)
            # Simplified: distribute to bottom 30% proportionally
            sorted_indices = np.argsort(wealth)
            n_eligible = int(0.3 * n_agents)
            eligible_indices = sorted_indices[:n_eligible]
            
            # Redistribution to eligible agents
            redistribution = np.zeros(n_agents)
            if n_eligible > 0:
                # Distribute based on need (inverse wealth)
                need = 1.0 / (wealth[eligible_indices] + 1)
                need = need / np.sum(need)
                redistribution[eligible_indices] = dist[0] * need + dist[1] * need / 2
            
            # Update wealth
            for i in range(n_agents):
                dW = self.wealth_dynamics(
                    wealth[i], t,
                    zakat_paid=zakat_per_agent[i],
                    zakat_received=redistribution[i]
                )
                wealth[i] = max(1, wealth[i] + dW * dt)
            
            # Store
            wealth_history[step] = wealth.copy()
            gini_history[step] = self.compute_gini(wealth)
        
        return {
            'wealth_history': wealth_history,
            'gini_history': gini_history,
            'zakat_collected': zakat_collected,
            'zakat_distributed': zakat_distributed,
            'zakat_by_category': zakat_by_category,
            't': np.arange(0, n_years, dt),
            'n_agents': n_agents,
            'initial_wealth': wealth_initial,
            'final_wealth': wealth
        }
    
    def compute_gini(self, wealth):
        """Compute Gini coefficient."""
        wealth = np.sort(wealth)
        n = len(wealth)
        cum_wealth = np.cumsum(wealth)
        total = cum_wealth[-1]
        
        if total == 0:
            return 0.0
        
        gini = (n + 1 - 2 * np.sum(cum_wealth) / total) / n
        return gini
    
    def compute_lorenz_curve(self, wealth):
        """Compute Lorenz curve."""
        wealth = np.sort(wealth)
        n = len(wealth)
        cum_wealth = np.cumsum(wealth)
        total = cum_wealth[-1]
        
        if total == 0:
            return np.linspace(0, 1, n), np.linspace(0, 1, n)
        
        pop_frac = np.arange(1, n + 1) / n
        wealth_frac = cum_wealth / total
        
        return pop_frac, wealth_frac
    
    def compute_metrics(self, results):
        """
        Compute key metrics from simulation results.
        """
        gini_history = results['gini_history']
        wealth_history = results['wealth_history']
        zakat_collected = results['zakat_collected']
        
        initial_gini = gini_history[0]
        final_gini = gini_history[-1]
        gini_reduction = (initial_gini - final_gini) / initial_gini
        
        total_zakat = np.sum(zakat_collected)
        avg_zakat = np.mean(zakat_collected)
        
        initial_wealth_mean = np.mean(wealth_history[0])
        final_wealth_mean = np.mean(wealth_history[-1])
        
        # Theil index (simplified)
        wealth = wealth_history[-1]
        wealth_mean = np.mean(wealth)
        theil = np.mean((wealth / wealth_mean) * np.log(wealth / wealth_mean + 1e-10))
        
        return {
            'initial_gini': initial_gini,
            'final_gini': final_gini,
            'gini_reduction': gini_reduction,
            'total_zakat': total_zakat,
            'avg_zakat': avg_zakat,
            'initial_wealth_mean': initial_wealth_mean,
            'final_wealth_mean': final_wealth_mean,
            'theil_index': theil
        }

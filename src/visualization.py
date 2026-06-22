"""
Visualization functions for the Zakat and Awqaf article.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from matplotlib.patches import Rectangle, Patch
from matplotlib.colors import LinearSegmentedColormap

# Set publication-ready style
plt.style.use('seaborn-v0-8-whitegrid')
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 14
rcParams['legend.fontsize'] = 10
rcParams['figure.dpi'] = 300

class FigureGenerator:
    """
    Generate figures for the Zakat and Awqaf article.
    """
    
    def __init__(self, output_dir='figures'):
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def figure_zakat_wealth_distribution(self, results, zakat_model):
        """
        Figure 1: Wealth distribution before and after Zakat.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        wealth_initial = results['initial_wealth']
        wealth_final = results['final_wealth']
        
        # Panel 1: Histograms
        ax = axes[0]
        bins = np.logspace(0, 4, 50)
        
        ax.hist(wealth_initial, bins=bins, alpha=0.5, color='blue', 
                label='Before Zakat', density=True)
        ax.hist(wealth_final, bins=bins, alpha=0.5, color='green',
                label='After Zakat', density=True)
        
        ax.set_xscale('log')
        ax.set_xlabel('Wealth', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Wealth Distribution', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Lorenz curves
        ax = axes[1]
        
        # Compute Lorenz curves
        pop_frac_init, wealth_frac_init = zakat_model.compute_lorenz_curve(wealth_initial)
        pop_frac_final, wealth_frac_final = zakat_model.compute_lorenz_curve(wealth_final)
        
        ax.plot(pop_frac_init, wealth_frac_init, linewidth=2.5, color='blue',
                label='Before Zakat')
        ax.plot(pop_frac_final, wealth_frac_final, linewidth=2.5, color='green',
                label='After Zakat')
        
        # Perfect equality line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Perfect equality')
        
        ax.set_xlabel('Cumulative population fraction', fontsize=12)
        ax.set_ylabel('Cumulative wealth fraction', fontsize=12)
        ax.set_title('Lorenz Curves', fontsize=14)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # Add Gini values
        gini_initial = zakat_model.compute_gini(wealth_initial)
        gini_final = zakat_model.compute_gini(wealth_final)
        
        ax.text(0.05, 0.10, f'$G_{{before}} = {gini_initial:.3f}$', 
                transform=ax.transAxes, fontsize=11)
        ax.text(0.05, 0.05, f'$G_{{after}} = {gini_final:.3f}$', 
                transform=ax.transAxes, fontsize=11)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/zakat_wealth_distribution.pdf', format='pdf')
        plt.savefig(f'{self.output_dir}/zakat_wealth_distribution.png', format='png', dpi=300)
        plt.close()
        
        return fig
    
    def figure_zakat_gini_reduction(self, results):
        """
        Figure 2: Gini coefficient reduction over time.
        """
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        t = results['t']
        gini_history = results['gini_history']
        
        # Panel 1: Gini over time
        ax = axes[0]
        ax.plot(t, gini_history, linewidth=2.5, color='darkblue')
        
        # Add horizontal lines
        ax.axhline(y=gini_history[0], color='red', linestyle='--', 
                   linewidth=1.5, label=f'Initial Gini: {gini_history[0]:.3f}')
        ax.axhline(y=gini_history[-1], color='green', linestyle='--', 
                   linewidth=1.5, label=f'Final Gini: {gini_history[-1]:.3f}')
        
        ax.set_xlabel('Time (years)', fontsize=12)
        ax.set_ylabel('Gini coefficient', fontsize=12)
        ax.set_title('Gini Coefficient Evolution with Zakat', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 0.6])
        
        # Panel 2: Gini reduction
        ax = axes[1]
        reduction = (gini_history[0] - gini_history) / gini_history[0]
        ax.plot(t, reduction * 100, linewidth=2.5, color='darkgreen')
        
        ax.set_xlabel('Time (years)', fontsize=12)
        ax.set_ylabel('Gini reduction (%)', fontsize=12)
        ax.set_title('Gini Coefficient Reduction', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Add final reduction value
        final_reduction = reduction[-1] * 100
        ax.axhline(y=final_reduction, color='red', linestyle='--', 
                   linewidth=1.5, label=f'Final reduction: {final_reduction:.1f}%')
        ax.legend(loc='lower right')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/zakat_gini_reduction.pdf', format='pdf')
        plt.savefig(f'{self.output_dir}/zakat_gini_reduction.png', format='png', dpi=300)
        plt.close()
        
        return fig
    
    def figure_waqf_dynamics(self, waqf_results):
        """
        Figure 3: Waqf stock and benefits evolution.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        t = waqf_results['t']
        W = waqf_results['W']
        B = waqf_results['B']
        Services = waqf_results['Services']
        S = waqf_results['S']
        D = waqf_results['D']
        
        # Panel 1: Waqf stock
        ax = axes[0, 0]
        ax.plot(t, W, linewidth=2.5, color='darkblue', label='Waqf stock $W(t)$')
        ax.set_xlabel('Time (years)', fontsize=12)
        ax.set_ylabel('Waqf stock', fontsize=12)
        ax.set_title('Waqf Stock Evolution', fontsize=14)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Benefits
        ax = axes[0, 1]
        ax.plot(t, B, linewidth=2.5, color='darkgreen', label='Benefits $B(t)$')
        ax.set_xlabel('Time (years)', fontsize=12)
        ax.set_ylabel('Benefits', fontsize=12)
        ax.set_title('Waqf Benefits Flow', fontsize=14)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Panel 3: Surplus and Donations
        ax = axes[1, 0]
        ax.plot(t, S, linewidth=2, color='orange', label='Surplus $S(t)$')
        ax.plot(t, D, linewidth=2, color='red', label='Donations $D(t)$')
        ax.set_xlabel('Time (years)', fontsize=12)
        ax.set_ylabel('Rate', fontsize=12)
        ax.set_title('Waqf Funding Sources', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Panel 4: Cumulative services
        ax = axes[1, 1]
        ax.plot(t, Services, linewidth=2.5, color='purple', 
                label='Cumulative Services')
        ax.set_xlabel('Time (years)', fontsize=12)
        ax.set_ylabel('Total services provided', fontsize=12)
        ax.set_title('Cumulative Social Services', fontsize=14)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/waqf_dynamics.pdf', format='pdf')
        plt.savefig(f'{self.output_dir}/waqf_dynamics.png', format='png', dpi=300)
        plt.close()
        
        return fig
    
    def figure_zakat_consumption_stabilization(self, results_with, results_without):
        """
        Figure 4: Consumption stabilization during shocks.
        """
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        t = results_with['t']
        
        # Panel 1: Consumption comparison
        ax = axes[0]
        ax.plot(t, results_with['consumption_history'], linewidth=2.5, 
                color='blue', label='With Zakat & Awqaf')
        ax.plot(t, results_without['consumption_history'], linewidth=2.5, 
                color='red', linestyle='--', label='Without Zakat & Awqaf')
        
        # Add shock indicators
        ax.axvline(x=20, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
        ax.axvline(x=25, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
        ax.text(22.5, ax.get_ylim()[1]*0.9, 'Shock period', 
                ha='center', fontsize=10, style='italic')
        
        ax.set_xlabel('Time (years)', fontsize=12)
        ax.set_ylabel('Consumption', fontsize=12)
        ax.set_title('Consumption Stabilization During Economic Shocks', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Volatility reduction
        ax = axes[1]
        
        # Compute rolling volatility
        window = 50
        vol_with = np.zeros(len(t) - window)
        vol_without = np.zeros(len(t) - window)
        
        for i in range(len(t) - window):
            vol_with[i] = np.std(results_with['consumption_history'][i:i+window])
            vol_without[i] = np.std(results_without['consumption_history'][i:i+window])
        
        vol_ratio = vol_with / vol_without
        t_vol = t[window:]
        
        ax.plot(t_vol, vol_ratio, linewidth=2.5, color='darkblue')
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5,
                   label='Equal volatility')
        ax.axhline(y=np.mean(vol_ratio), color='green', linestyle='--', 
                   linewidth=1.5, label=f'Mean ratio: {np.mean(vol_ratio):.2f}')
        
        ax.set_xlabel('Time (years)', fontsize=12)
        ax.set_ylabel('Volatility ratio (with / without)', fontsize=12)
        ax.set_title('Consumption Volatility Reduction', fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.2])
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/zakat_consumption_stabilization.pdf', format='pdf')
        plt.savefig(f'{self.output_dir}/zakat_consumption_stabilization.png', format='png', dpi=300)
        plt.close()
        
        return fig
    
    def figure_full_system_comparison(self, results_systems, labels=None):
        """
        Figure 5: Full system comparison.
        """
        if labels is None:
            labels = ['Without Zakat/Awqaf', 'With Zakat only', 
                     'With Waqf only', 'With Zakat & Awqaf']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        colors = ['red', 'orange', 'green', 'blue']
        
        # Panel 1: Gini coefficient
        ax = axes[0, 0]
        for i, (label, results) in enumerate(zip(labels, results_systems)):
            ax.plot(results['t'], results['gini_history'], 
                    linewidth=2, color=colors[i], label=label)
        ax.set_xlabel('Time (years)', fontsize=12)
        ax.set_ylabel('Gini coefficient', fontsize=12)
        ax.set_title('Gini Coefficient Comparison', fontsize=14)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 0.6])
        
        # Panel 2: Consumption
        ax = axes[0, 1]
        for i, (label, results) in enumerate(zip(labels, results_systems)):
            ax.plot(results['t'], results['consumption_history'], 
                    linewidth=2, color=colors[i], label=label)
        ax.set_xlabel('Time (years)', fontsize=12)
        ax.set_ylabel('Consumption', fontsize=12)
        ax.set_title('Consumption Comparison', fontsize=14)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Panel 3: Social services
        ax = axes[1, 0]
        for i, (label, results) in enumerate(zip(labels, results_systems)):
            if 'social_services' in results:
                ax.plot(results['t'], results['social_services'], 
                        linewidth=2, color=colors[i], label=label)
        ax.set_xlabel('Time (years)', fontsize=12)
        ax.set_ylabel('Social services', fontsize=12)
        ax.set_title('Social Services Provision', fontsize=14)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Panel 4: Summary metrics (bar chart)
        ax = axes[1, 1]
        
        metrics = ['Gini\nreduction', 'Consumption\nstability', 'Social\nservices']
        x = np.arange(len(metrics))
        width = 0.2
        
        for i, (label, results) in enumerate(zip(labels, results_systems)):
            # Compute metrics
            gini_reduction = (results['gini_history'][0] - results['gini_history'][-1]) / results['gini_history'][0]
            consumption_std = np.std(results['consumption_history'])
            social_total = np.sum(results['social_services']) if 'social_services' in results else 0
            
            # Normalize
            if i == 0:
                gini_reduction_norm = 0
                consumption_std_norm = 1
                social_total_norm = 0
            else:
                gini_reduction_norm = gini_reduction / max([r['gini_history'][0] - r['gini_history'][-1] for r in results_systems])
                consumption_std_norm = 1 - consumption_std / max([np.std(r['consumption_history']) for r in results_systems])
                social_total_norm = social_total / max([np.sum(r['social_services']) if 'social_services' in r else 1 for r in results_systems])
            
            values = [gini_reduction_norm, consumption_std_norm, social_total_norm]
            ax.bar(x + i*width, values, width, color=colors[i], label=label, alpha=0.7)
        
        ax.set_xlabel('Metric', fontsize=12)
        ax.set_ylabel('Normalized score', fontsize=12)
        ax.set_title('System Performance Summary', fontsize=14)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(metrics)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/full_system_comparison.pdf', format='pdf')
        plt.savefig(f'{self.output_dir}/full_system_comparison.png', format='png', dpi=300)
        plt.close()
        
        return fig
    
    def figure_all(self, results_zakat, waqf_results, 
                   results_systems, labels):
        """
        Generate all figures for the article.
        """
        print("Generating Figure 1: Wealth distribution...")
        self.figure_zakat_wealth_distribution(results_zakat, ZakatModel())
        
        print("Generating Figure 2: Gini reduction...")
        self.figure_zakat_gini_reduction(results_zakat)
        
        print("Generating Figure 3: Waqf dynamics...")
        self.figure_waqf_dynamics(waqf_results)
        
        print("Generating Figure 4: Consumption stabilization...")
        # Assuming results_with and results_without are first two
        self.figure_zakat_consumption_stabilization(
            results_systems[3], results_systems[0]
        )
        
        print("Generating Figure 5: Full system comparison...")
        self.figure_full_system_comparison(results_systems, labels)
        
        print("All figures generated successfully!")

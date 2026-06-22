#!/usr/bin/env python3
"""
Run Zakat and Awqaf simulations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pickle
from src.zakat_model import ZakatModel
from src.waqf_model import WaqfModel
from src.social_stabilization import SocialStabilizationModel
from src.visualization import FigureGenerator

def run_zakat_simulation():
    """
    Run Zakat simulation.
    """
    print("=" * 60)
    print("Running Zakat Simulation...")
    print("=" * 60)
    
    # Initialize model
    zakat_model = ZakatModel(tau=0.025)
    
    # Generate initial wealth distribution
    wealth_initial = zakat_model.generate_wealth_distribution(
        n_agents=10000, method='lognormal'
    )
    
    # Run simulation
    results = zakat_model.simulate_zakat(
        wealth_initial, n_years=50, dt=0.1
    )
    
    # Compute metrics
    metrics = zakat_model.compute_metrics(results)
    
    print("\nZakat Simulation Results:")
    print(f"  Initial Gini: {metrics['initial_gini']:.3f}")
    print(f"  Final Gini: {metrics['final_gini']:.3f}")
    print(f"  Gini reduction: {metrics['gini_reduction']:.1%}")
    print(f"  Total Zakat collected: {metrics['total_zakat']:.2f}")
    print(f"  Average Zakat: {metrics['avg_zakat']:.2f}")
    print(f"  Theil index: {metrics['theil_index']:.3f}")
    
    # Save results
    os.makedirs('data', exist_ok=True)
    with open('data/results_zakat.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    return results, zakat_model

def run_waqf_simulation():
    """
    Run Waqf simulation.
    """
    print("\n" + "=" * 60)
    print("Running Waqf Simulation...")
    print("=" * 60)
    
    # Initialize model
    waqf_model = WaqfModel(
        phi=0.05, psi=0.02, omega=0.08, chi=0.03,
        S_base=100.0, donation_rate=0.01
    )
    
    # Run simulation
    results = waqf_model.simulate_waqf(
        W0=500.0, B0=50.0, n_years=50, dt=0.1
    )
    
    # Compute metrics
    metrics = waqf_model.compute_metrics(results)
    
    print("\nWaqf Simulation Results:")
    print(f"  Final Waqf stock: {metrics['final_waqf']:.2f}")
    print(f"  Final benefits: {metrics['final_benefits']:.2f}")
    print(f"  Total services: {metrics['total_services']:.2f}")
    print(f"  Waqf growth: {metrics['waqf_growth']:.1%}")
    
    # Save results
    with open('data/results_waqf.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    return results, waqf_model

def run_full_system_simulation():
    """
    Run full system simulation with different configurations.
    """
    print("\n" + "=" * 60)
    print("Running Full System Simulations...")
    print("=" * 60)
    
    # Initialize
    zakat_model = ZakatModel()
    waqf_model = WaqfModel()
    model = SocialStabilizationModel(zakat_model, waqf_model)
    
    # Generate initial wealth
    wealth_initial = zakat_model.generate_wealth_distribution(
        n_agents=10000, method='lognormal'
    )
    
    # Run different configurations
    configs = [
        {'with_zakat': False, 'with_waqf': False},
        {'with_zakat': True, 'with_waqf': False},
        {'with_zakat': False, 'with_waqf': True},
        {'with_zakat': True, 'with_waqf': True}
    ]
    
    labels = ['Without Zakat/Awqaf', 'With Zakat only', 
              'With Waqf only', 'With Zakat & Awqaf']
    
    results_systems = []
    
    for config in configs:
        print(f"\n  Running: {config}")
        results = model.simulate_full_system(
            wealth_initial, n_years=50, dt=0.1,
            with_zakat=config['with_zakat'],
            with_waqf=config['with_waqf']
        )
        results_systems.append(results)
        
        print(f"    Final Gini: {results['gini_history'][-1]:.3f}")
    
    # Save results
    with open('data/results_systems.pkl', 'wb') as f:
        pickle.dump(results_systems, f)
    with open('data/system_labels.pkl', 'wb') as f:
        pickle.dump(labels, f)
    
    return results_systems, labels

def main():
    """
    Main execution function.
    """
    # Run simulations
    results_zakat, zakat_model = run_zakat_simulation()
    results_waqf, waqf_model = run_waqf_simulation()
    results_systems, labels = run_full_system_simulation()
    
    print("\n" + "=" * 60)
    print("All simulations complete. Results saved.")
    print("=" * 60)
    
    # Generate figures
    print("\nGenerating figures...")
    fig_gen = FigureGenerator()
    fig_gen.figure_all(results_zakat, results_waqf, results_systems, labels)
    
    print("\nDone!")

if __name__ == "__main__":
    main()

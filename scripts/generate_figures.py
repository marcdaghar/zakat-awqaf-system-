#!/usr/bin/env python3
"""
Generate all figures from saved results.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
from src.zakat_model import ZakatModel
from src.visualization import FigureGenerator

def main():
    """
    Load saved results and generate figures.
    """
    print("Loading saved results...")
    
    with open('data/results_zakat.pkl', 'rb') as f:
        results_zakat = pickle.load(f)
    
    with open('data/results_waqf.pkl', 'rb') as f:
        results_waqf = pickle.load(f)
    
    with open('data/results_systems.pkl', 'rb') as f:
        results_systems = pickle.load(f)
    
    with open('data/system_labels.pkl', 'rb') as f:
        labels = pickle.load(f)
    
    zakat_model = ZakatModel()
    
    print("Generating figures...")
    fig_gen = FigureGenerator()
    fig_gen.figure_all(results_zakat, results_waqf, results_systems, labels)
    
    print("All figures generated successfully!")

if __name__ == "__main__":
    main()

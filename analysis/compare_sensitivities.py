import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from pricing_rfv import VasicekRFV

# Plotting Style
plt.style.use('seaborn-v0_8-whitegrid')

def calculate_rmv_duration(kappa, T, L, lambda_1):
    """
    Calculates analytical Duration for RMV.
    D_RMV = (1 + L * lambda_1) * B(kappa, T)
    """
    B = (1 - np.exp(-kappa * T)) / kappa
    k1 = 1 + L * lambda_1
    return k1 * B

def calculate_rmv_convexity(kappa, T, L, lambda_1):
    """
    Calculates analytical Convexity for RMV.
    C_RMV = ((1 + L * lambda_1) * B(kappa, T))^2
    """
    B = (1 - np.exp(-kappa * T)) / kappa
    k1 = 1 + L * lambda_1
    return (k1 * B)**2

def main():
    # Parameters (Table 5.1)
    r0 = 0.04
    kappa = 0.15
    sigma = 0.01
    theta = 0.0522
    L = 0.4
    lambda_0 = 0.025
    T = 10
    
    # Lambda_1 Range
    lambda_1_values = np.linspace(-0.10, 0.10, 50)
    
    # Store results
    results = {
        'lambda_1': lambda_1_values,
        'dur_rmv': [], 'dur_rfv': [],
        'conv_rmv': [], 'conv_rfv': []
    }
    
    print("Starting Sensitivity Comparison Loop...")
    
    for l1 in lambda_1_values:
        # 1. RMV Analytical
        d_rmv = calculate_rmv_duration(kappa, T, L, l1)
        c_rmv = calculate_rmv_convexity(kappa, T, L, l1)
        
        results['dur_rmv'].append(d_rmv)
        results['conv_rmv'].append(c_rmv)
        
        # 2. RFV Numerical
        model = VasicekRFV(r0, kappa, sigma, theta, lambda_0, l1, L)
        d_rfv = model.duration(0, T)
        c_rfv = model.convexity(0, T)
        
        results['dur_rfv'].append(d_rfv)
        results['conv_rfv'].append(c_rfv)
        
    # Convert to arrays
    for key in results:
        results[key] = np.array(results[key])
        
    # --- Plot 1: Duration (Existing) ---
    rel_error_dur = 100 * (results['dur_rfv'] - results['dur_rmv']) / results['dur_rfv']
    
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    ax1.plot(lambda_1_values, results['dur_rmv'], label='RMV (Analytical)', linewidth=2, linestyle='--')
    ax1.plot(lambda_1_values, results['dur_rfv'], label='RFV (New Model)', linewidth=2)
    ax1.set_title('Stochastic Duration: RMV vs RFV', fontsize=14)
    ax1.set_ylabel('Duration (Years)', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(lambda_1_values, rel_error_dur, color='red', linewidth=2)
    ax2.set_title('Relative Difference (RFV - RMV) / RFV', fontsize=14)
    ax2.set_xlabel('Lambda_1 (Sensitivity to Interest Rate)', fontsize=12)
    ax2.set_ylabel('Difference (%)', fontsize=12)
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    output_path_dur = os.path.join(os.path.dirname(__file__), '..', 'figures', 'duration_comparison.png')
    plt.tight_layout()
    plt.savefig(output_path_dur, dpi=300)
    print(f"Duration Figure saved to: {output_path_dur}")
    
    # --- Plot 2: Convexity (New) ---
    diff_conv = results['conv_rfv'] - results['conv_rmv']
    
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Top: Absolute Levels
    ax3.plot(lambda_1_values, results['conv_rmv'], label='RMV (Analytical)', linewidth=2, linestyle='--')
    ax3.plot(lambda_1_values, results['conv_rfv'], label='RFV (New Model)', linewidth=2)
    ax3.set_title('Stochastic Convexity: RMV vs RFV', fontsize=14)
    ax3.set_ylabel('Convexity', fontsize=12)
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Bottom: Difference
    ax4.plot(lambda_1_values, diff_conv, color='purple', linewidth=2)
    ax4.set_title('Difference (RFV - RMV)', fontsize=14)
    ax4.set_xlabel('Lambda_1 (Sensitivity to Interest Rate)', fontsize=12)
    ax4.set_ylabel('Difference (Absolute)', fontsize=12)
    ax4.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax4.grid(True, alpha=0.3)
    
    output_path_conv = os.path.join(os.path.dirname(__file__), '..', 'figures', 'convexity_comparison.png')
    plt.tight_layout()
    plt.savefig(output_path_conv, dpi=300)
    print(f"Convexity Figure saved to: {output_path_conv}")

if __name__ == "__main__":
    main()

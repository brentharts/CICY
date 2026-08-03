import os, sys, argparse, ast
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyCICY import CICY
from pyCICY.smoothness import is_smooth

# Standard Model known constants
SM_MU_RATIO = 1836.152673  # m_p / m_e
SM_ALPHA_INV = 137.035999  # Fine-structure constant inverse

def compute_mass_ratio(conf_matrix):
    """
    PLACEHOLDER: First-principles string theory mass ratio calculation.
    Currently, this returns a mock value based on the manifold's dimension 
    and configuration shape, as exact mass derivation from topology alone 
    is an unsolved problem in string phenomenology.
    """
    # Convert to numpy array to extract basic properties
    M = np.array(conf_matrix, dtype=np.int16)
    dim_A = sum([M[i][0] for i in range(len(M))])
    
    # Mock calculation: generating a pseudo-random ratio based on the matrix hash
    # Replace this with your Type II analytical approximations later.
    pseudo_val = abs(hash(str(conf_matrix))) % 500
    predicted_ratio = 1500 + pseudo_val 
    
    return predicted_ratio

def analyze_topology(conf_matrix):
    """
    Uses pyCICY to run heavy topological calculations.
    """
    print("\n--- Running Heavy Topological Analysis ---")
    
    # Initialize the CICY object
    M = CICY(conf_matrix, log=3)
    
    if not M.CY:
        print("Warning: This configuration does not belong to a Calabi-Yau manifold.")
        
    print(f"Manifold Fold: {M.nfold}-fold")
    
    # pyCICY supports Hodge data for 2-, 3-, and 4-folds
    if 2 <= M.nfold <= 4:
        print(f"Euler Characteristic: {M.euler}")
        print(f"Hodge Numbers: {M.h}")
        if M.nfold == 3:
            print(f"Favourable: {M.fav}")
    else:
        print("Hodge data is only supported for 2-, 3-, and 4-folds.")
        
    return M

def plot_ratio_comparison(predicted_mu):
    """
    Plots the predicted mass ratio against the real Standard Model value.
    """
    labels = ['Standard Model', 'String Theory (Type II CY Prediction)']
    values = [SM_MU_RATIO, predicted_mu]
    colors = ['blue', 'orange']

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, values, color=colors, alpha=0.7)
    
    # Add a dashed line for the target Standard Model value
    ax.axhline(y=SM_MU_RATIO, color='r', linestyle='--', label=f'Target: {SM_MU_RATIO:.2f}')
    
    ax.set_ylabel(r'Proton-to-Electron Mass Ratio ($\mu = m_p/m_e$)')
    ax.set_title('CY Configuration vs. Standard Model')
    ax.set_ylim(0, max(SM_MU_RATIO, predicted_mu) * 1.2)
    ax.legend()

    # Label the bars with exact values
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Analyze CICY configurations for Standard Model phenomenological properties.")
    parser.add_argument("--cicy", type=str, required=True,
                        help="CICY configuration matrix as a string, e.g., '[[4, 5]]' for the quintic.")
    parser.add_argument("--proton-electron-mass-ratio", action="store_true",
                        help="Fast execution: Skips heavy topological checks and only predicts the mass ratio.")
    
    args = parser.parse_args()
    
    # Parse the configuration matrix securely
    try:
        conf_matrix = ast.literal_eval(args.cicy)
    except (ValueError, SyntaxError):
        print("Error: Invalid configuration matrix format. Use format '[[4, 5]]'")
        return

    print(f"Input Configuration: {conf_matrix}")

    # Fast track: Only compute mass ratio
    if args.proton_electron_mass_ratio:
        print("\n[Fast Mode Enabled]: Skipping heavy topology calculations.")
        predicted_mu = compute_mass_ratio(conf_matrix)
        print(f"Predicted mass ratio (mu): {predicted_mu:.4f}")
        plot_ratio_comparison(predicted_mu)
        return

    # Standard track: Run pyCICY topology checks first
    M = analyze_topology(conf_matrix)
    
    # Calculate mass ratio
    print("\n--- Computing Phenomenological Ratios ---")
    predicted_mu = compute_mass_ratio(conf_matrix)
    print(f"Predicted mass ratio (mu): {predicted_mu:.4f}")
    
    # Plot results
    plot_ratio_comparison(predicted_mu)

if __name__ == "__main__":
    main()

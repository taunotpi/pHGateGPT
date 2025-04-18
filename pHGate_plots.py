"""
Created on March 31, 2025

@author: samlevy

pH Gating Activation Visualization Script
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_gating_activations(gating_activations, labels=None):
    """
    Plots the gating activations with a heatmap.

    Args:
    - gating_activations (np.array or list of list): The gating activations to plot.
    - labels (list, optional): The labels for the gating activations.

    Returns:
    - A plot that isn't a purple square
    """
    if labels is None:
        labels = list(range(len(gating_activations)))

    plt.figure(figsize=(12, 8))
    sns.heatmap(gating_activations, annot=True, fmt=".2f", cmap='viridis', xticklabels=labels, yticklabels=labels)
    plt.title('Heatmap of Gating Activations')
    plt.xlabel('Gating Activation Index')
    plt.ylabel('Gating Activation Index')
    plt.show()


# Example usage:
if __name__ == "__main__":

    # Example data for gating activations
    gating_activations = np.random.rand(10, 10)  # Replace with actual gating activations data
    plot_gating_activations(gating_activations)


# Simulated example: Gating activation values for 10 HA sequences
np.random.seed(42)
gating_scores = np.random.rand(10)

# Simulated example: Known pH activation thresholds for the same 10 sequences
# Lower values mean they activate at lower pH (more infectious)
ph_thresholds = np.array([5.0, 5.2, 5.5, 4.8, 5.1, 6.0, 5.8, 5.3, 5.0, 5.6])

# Create a DataFrame for visualization
df = pd.DataFrame({
    'Gating Score': gating_scores,
    'HA pH Activation Threshold': ph_thresholds
})

# Plot: Correlation between gating activation and pH activation threshold
plt.figure(figsize=(10, 6))
sns.regplot(x='HA pH Activation Threshold', y='Gating Score', data=df, marker='o', scatter_kws={"s": 80})
plt.title('Correlation between pH Threshold and Gating Score')
plt.xlabel('HA pH Activation Threshold (lower = more likely to fuse)')
plt.ylabel('Gating Activation Score')
plt.grid(True)
plt.tight_layout()
plt.show()

# Compute and return the correlation coefficient
correlation = np.corrcoef(gating_scores, ph_thresholds)[0, 1]

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the models and feature selection methods
models = [
    'LightGBM', 'LightGBMXT', 'CatBoost', 'XGBoost',
    'RandomForestGini', 'RandomForestEntr', 'ExtraTreesGini', 'ExtraTreesEntr',
    'KNeighborsUnif', 'KNeighborsDist'
]

methods = ['shap', 'mrmr', 'anova']

# Output directory where the results are stored
output_dir = '../ML_results'

# Initialize DataFrames to store the highest ACC and MCC for each model and method
acc_df = pd.DataFrame(index=models, columns=methods)
mcc_df = pd.DataFrame(index=models, columns=methods)

# Loop through each model and method to extract the highest ACC and MCC
for model_name in models:
    for method in methods:
        highest_acc = 0
        highest_mcc = -1  # MCC can be negative
        results_dir = os.path.join(output_dir, model_name, method)
        if os.path.exists(results_dir):
            # Get all result files in the directory
            results_files = [
                f for f in os.listdir(results_dir)
                if f.startswith('results_') and f.endswith('.txt')
            ]
            for result_file in results_files:
                result_path = os.path.join(results_dir, result_file)
                with open(result_path, 'r') as f:
                    content = f.read()
                    # Extract 'Accuracies per fold' and 'MCCs per fold' using regular expressions
                    accuracies_match = re.search(
                        r'Accuracies per fold:\s*(\[[^\]]*\])', content)
                    mccs_match = re.search(
                        r'MCCs per fold:\s*(\[[^\]]*\])', content)
                    if accuracies_match:
                        accuracies_str = accuracies_match.group(1)
                        accuracies_list = eval(accuracies_str)
                        max_acc = max(accuracies_list)
                        if max_acc > highest_acc:
                            highest_acc = max_acc
                    if mccs_match:
                        mccs_str = mccs_match.group(1)
                        mccs_list = eval(mccs_str)
                        max_mcc = max(mccs_list)
                        if max_mcc > highest_mcc:
                            highest_mcc = max_mcc
        else:
            print(
                f"No results directory for model '{model_name}' and method '{method}'."
            )
        acc_df.loc[model_name, method] = highest_acc
        mcc_df.loc[model_name, method] = highest_mcc

# Replace any missing values with NaN
acc_df = acc_df.apply(pd.to_numeric, errors='coerce')
mcc_df = mcc_df.apply(pd.to_numeric, errors='coerce')

# Plotting configuration
metrics = ['ACC', 'MCC']
dataframes = [acc_df, mcc_df]
method_styles = {
    'shap': {'color': '#FFCC00', 'linestyle': '-'},#FFCC00 #009999 #CC3366
    'mrmr': {'color': '#009999', 'linestyle': '-'},
    'anova': {'color': '#CC3366', 'linestyle': '-'}
}
# Create output directory for plots
output_plots_dir = 'ML_plot'
os.makedirs(output_plots_dir, exist_ok=True)

# Create a mapping for method names to the legend labels
method_labels = {
    'shap': 'SHAP',
    'mrmr': 'mRMR',
    'anova': 'ANOVA'
}

# Plot both ACC and MCC
x = np.arange(len(models))  # x positions
fig, axes = plt.subplots(1, 2, figsize=(32, 6))

for i, (metric, df) in enumerate(zip(metrics, dataframes)):
    ax = axes[i]
    for method in methods:
        y = df[method].values
        ax.plot(
            x,
            y,
            label=method_labels[method],  # Use the custom legend labels here
            **method_styles[method],
            linewidth=4
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=50, ha='center', fontsize=25, fontweight='bold')
    ax.set_ylabel(metric, fontsize=30, fontweight='bold', fontstyle='italic')

    # Adjust legend position by moving it to the right
    legend = ax.legend(fontsize=30, loc='upper right', frameon=False, bbox_to_anchor=(1.1, 1.1))
    for text in legend.get_texts():
        text.set_fontweight('bold')
    for line in legend.get_lines():
        line.set_linewidth(8)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params(axis='y', labelsize=30)  # Increase the y-axis tick label size
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight('bold')

plt.tight_layout()
plt.savefig(os.path.join(output_plots_dir, 'ML_ACCMCC.svg'))

import os
import copy
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, matthews_corrcoef, confusion_matrix, average_precision_score
from sklearn.metrics import roc_curve, precision_recall_curve, auc as sklearn_auc
import matplotlib.pyplot as plt




# Define the select_features function
def select_features(method, X, y, shap_features=None, mrmr_features=None, anova_features=None):
    if method == 'shap':
        if shap_features is None:
            raise ValueError("shap_features must be provided for shap method.")
        selected_features = [feature for feature in shap_features if feature in X.columns]
        return selected_features
    elif method == 'mrmr':
        if mrmr_features is None:
            raise ValueError("mrmr_features must be provided for mRMR method.")
        selected_features = [feature for feature in mrmr_features if feature in X.columns]
        return selected_features
    elif method == 'anova':
        if anova_features is None:
            raise ValueError("anova_features must be provided for anova method.")
        selected_features = [feature for feature in anova_features if feature in X.columns]
        return selected_features
    else:
        raise ValueError(f"Unknown feature selection method: {method}")

# Load data
data = pd.read_csv('../Data/data.csv')

# Extract features and target
X = data.drop(columns=['target'])
y = data['target']

# Read SHAP, mRMR, and ANOVA selected features
shap_features_df = pd.read_csv('../Data/shap_features.csv')
shap_features = shap_features_df.iloc[:, 0].tolist()

mrmr_features_df = pd.read_csv('../Data/mrmr_features.csv')
mrmr_features = mrmr_features_df.iloc[:, 0].tolist()

anova_features_df = pd.read_csv('../Data/anova_features.csv')
anova_features = anova_features_df.iloc[:, 0].tolist()

class NonTransformerModel(nn.Module):
    def __init__(self, input_dim):
        super(NonTransformerModel, self).__init__()
        # A simple feed-forward network without the transformer layer
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.fc(x)
        return x  # Output logits directly

class NonAttentionModel(nn.Module):
    def __init__(self, input_dim, d_model=64, dropout=0.2):
        super(NonAttentionModel, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # Fully connected layers
        self.fc1 = nn.Linear(input_dim, d_model)  # First projection layer
        self.fc2 = nn.Linear(d_model, input_dim)  # Second projection layer

        # Output layer for binary classification
        self.fc_out = nn.Linear(input_dim, 1)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        # Apply fully connected layers
        x = self.fc1(x)  # (batch_size, d_model)
        x = torch.relu(x)  # Apply ReLU activation

        x = self.fc2(x)  # (batch_size, input_dim)
        x = torch.relu(x)  # Apply ReLU activation

        # Apply dropout
        x = self.dropout(x)

        # Classification layer
        x = self.fc_out(x)  # (batch_size, 1)
        return x  # Output logits directly

# Initialize models
models_dict = {
    'NoAttention': NonAttentionModel,
    'NonTransformer': NonTransformerModel,
}

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to parse results files and extract per-fold AUCs
def parse_results_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    num_features = None
    avg_auc = None
    aucs_per_fold = []
    for line in lines:
        if line.startswith('Number of Features:'):
            num_features = int(line.strip().split(':')[1])
        elif line.startswith('Average AUC:'):
            avg_auc = float(line.strip().split(':')[1])
        elif line.startswith('AUCs per fold:'):
            aucs_str = line.strip().split(':')[1].strip()
            aucs_str = aucs_str.strip('[]')
            aucs_per_fold = [float(x.strip()) for x in aucs_str.split(',')]
    return num_features, avg_auc, aucs_per_fold

# Prepare to collect plot data
plot_data = {}

# Models and methods to consider
model_names = ['NoAttention', 'NoTransformer']
methods = ['shap', 'mrmr', 'anova']
output_dir = '../AEno_results'

for model_name in model_names:
    print(f"Processing model: {model_name}")
    plot_data[model_name] = {'roc': {}, 'prc': {}}
    for method in methods:
        print(f"  Using feature selection method: {method.upper()}")
        results_dir = os.path.join(output_dir, model_name, method)
        if not os.path.exists(results_dir):
            print(f"    Results directory does not exist: {results_dir}")
            continue

        # Find the best configuration based on the highest single-fold AUC
        result_files = [f for f in os.listdir(results_dir) if f.startswith('results_') and f.endswith('.txt')]
        best_single_fold_auc = -1
        best_i = None
        best_fold_idx = None
        for result_file in result_files:
            filepath = os.path.join(results_dir, result_file)
            num_features, avg_auc, aucs_per_fold = parse_results_file(filepath)
            if num_features is None or not aucs_per_fold:
                continue
            for fold_idx, fold_auc in enumerate(aucs_per_fold):  # Changed 'auc' to 'fold_auc'
                if fold_auc > best_single_fold_auc:
                    best_single_fold_auc = fold_auc
                    best_i = num_features
                    best_fold_idx = fold_idx  # zero-based index

        if best_i is None or best_fold_idx is None:
            print(f"    No valid results found for method {method} in model {model_name}")
            continue

        print(f"    Best number of features: {best_i}, Best Single-Fold AUC: {best_single_fold_auc}, Fold: {best_fold_idx+1}")

        # Select features
        sorted_features = select_features(method, X, y, shap_features, mrmr_features, anova_features)
        current_features = sorted_features[:best_i]
        X_selected = X[current_features]

        # Prepare tensors
        X_tensor = torch.tensor(X_selected.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

        # Set up KFold cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(kf.split(X_tensor))

        # Get the validation data for the best fold
        train_index, val_index = splits[best_fold_idx]
        X_val_fold = X_tensor[val_index]
        y_val_fold = y_tensor[val_index]

        # Move tensors to device
        X_val_fold = X_val_fold.to(device)
        y_val_fold = y_val_fold.to(device)

        # Load the model for this fold
        model_save_dir = os.path.join(
            output_dir,
            model_name,
            method,
            f'num_features_{best_i}',
            'best_fold_auc',
            f'fold_{best_fold_idx+1}'
        )
        model_save_path = os.path.join(model_save_dir, 'model.pth')
        if not os.path.exists(model_save_path):
            print(f"    Model file does not exist: {model_save_path}")
            continue

        # Initialize the model
        input_dim = len(current_features)
        model_class = models_dict[model_name]
        model = model_class(input_dim).to(device)

        # Load the model state dict
        model.load_state_dict(torch.load(model_save_path, map_location=device))

        # Evaluation phase
        with torch.no_grad():
            model.eval()
            y_pred_logits = model(X_val_fold)
            y_pred_probs = torch.sigmoid(y_pred_logits).cpu().numpy().flatten()
            y_true = y_val_fold.cpu().numpy().astype(int).flatten()

        # Compute ROC and PRC curves
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
        roc_auc_value = sklearn_auc(fpr, tpr)  # Use 'sklearn_auc' instead of 'auc'

        precision, recall, thresholds_prc = precision_recall_curve(y_true, y_pred_probs)
        prc_auc_value = sklearn_auc(recall, precision)  # Use 'sklearn_auc' instead of 'auc'
        average_precision = average_precision_score(y_true, y_pred_probs)

        # Store the plot data
        plot_data[model_name]['roc'][method] = (fpr, tpr, roc_auc_value)
        plot_data[model_name]['prc'][method] = (recall, precision, average_precision)

# Now plot the ROC and PRC curves and save them to a folder
output_plots_dir = 'AEno_plot'
os.makedirs(output_plots_dir, exist_ok=True)

# Prepare the figure
fig, axes = plt.subplots(2, 2, figsize=(16, 14))  # 2 rows, 2 columns
plt.subplots_adjust(wspace=0.2, hspace=0.3)
axes = axes.flatten()

# Model names in order
model_order = ['NoAttention', 'NoTransformer']
plot_types = ['roc', 'prc']
line_colors = ['#FFCC00', '#009999', '#CC3366']

# Mapping from index to model and plot type
for idx in range(len(model_order) * len(plot_types)):
    model_idx = idx // len(plot_types)
    plot_type_idx = idx % len(plot_types)
    model_name = model_order[model_idx]
    plot_type = plot_types[plot_type_idx]
    ax = axes[idx]

    for method_idx, method in enumerate(methods):
        if method not in plot_data[model_name][plot_type]:
            continue
        color = line_colors[method_idx % len(line_colors)]
        method_label = method.upper() if method != 'mrmr' else 'mRMR'

        if plot_type == 'roc':
            fpr, tpr, auc_score = plot_data[model_name]['roc'][method]
            ax.plot(fpr, tpr, label=f"{method_label} (AUC={auc_score:.4f})", linewidth=4, color=color)
        elif plot_type == 'prc':
            recall, precision, avg_prec = plot_data[model_name]['prc'][method]
            ax.plot(recall, precision, label=f"{method_label} (AP={avg_prec:.4f})", linewidth=4, color=color)

    fontsize = 28
    if plot_type == 'roc':
        ax.set_xlabel('False Positive Rate', fontsize=fontsize, fontstyle='italic', fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=fontsize, fontstyle='italic', fontweight='bold')
        ax.set_title(f'{model_name} - ROC', fontsize=30, fontweight='bold')
    else:
        ax.set_xlabel('Recall', fontsize=fontsize, fontstyle='italic', fontweight='bold')
        ax.set_ylabel('Precision', fontsize=fontsize, fontstyle='italic', fontweight='bold')
        ax.set_title(f'{model_name} - PRC', fontsize=30, fontweight='bold')

    ax.tick_params(axis='both', which='major', labelsize=fontsize, width=2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    legend = ax.legend(fontsize=26, loc='lower center', frameon=False)
    for text in legend.get_texts():
        text.set_fontweight('bold')
    for line in legend.get_lines():
        line.set_linewidth(8)

# Save the figure
plot_save_path = os.path.join(output_plots_dir, 'AEno_ROCPRC.svg')
plt.tight_layout()
plt.savefig(plot_save_path)
plt.close()
print(f"ROC and PRC curves saved at {plot_save_path}")

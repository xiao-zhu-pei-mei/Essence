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
import umap
from matplotlib.colors import ListedColormap
plt.rcParams.update({'font.weight': 'bold'})

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

# Prepare to collect UMAP data
umap_data = {}

# Models and methods to consider
model_names = ['NoAttention', 'NoTransformer']
methods = ['shap', 'mrmr', 'anova']
output_dir = '../AEno_results'

for method in methods:
    print(f"Processing feature selection method: {method.upper()}")
    umap_data[method] = {}
    for model_name in model_names:
        print(f"  Processing model: {model_name}")
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

        # Set model to evaluation mode
        model.eval()

        # Extract internal representations
        with torch.no_grad():
            # For different models, we might need to adjust how we extract the internal features
            # Let's assume that we can modify the models to output the features we need
            # For simplicity, we'll define a function that wraps the model to get the desired output
            def get_internal_representation(model, x):
                # For SimpleClassificationNetwork (NoAttention)
                if isinstance(model, NonAttentionModel):
                    # Get the output after the first fully connected layer (before output layer)
                    x = model.fc1(x)  # Output from first fully connected layer
                    x = F.relu(x)  # Apply ReLU activation
                    return x.cpu().numpy()
                # For NoTransformerModel
                elif isinstance(model, NonTransformerModel):
                    # Since it's a simple linear model, return the output directly
                    x = model.fc(x)  # Output of the linear layer
                    return x.cpu().numpy()
                else:
                    # For other models (in case more models are used in the future)
                    return x.cpu().numpy()


            # Get internal representations
            internal_features = get_internal_representation(model, X_val_fold)

            # Store the UMAP data
            umap_data[method][model_name] = {
                'features': internal_features,
                'labels': y_val_fold.cpu().numpy().flatten()
            }

# Updated UMAP plot generation
# Prepare the figure
fig, axes = plt.subplots(3, 2, figsize=(16, 21))  # 3 rows, 2 columns (since 2 models and 3 methods)
plt.subplots_adjust(left=0.04, right=0.9, top=0.95, bottom=0.05, wspace=0.18, hspace=0.25)
axes = axes.flatten()  # Flatten axes to make it easier to iterate

# Increase the border width for all plots
for ax in axes:
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)

# Define output directory for plots
output_plots_dir = 'AEno_plot'
os.makedirs(output_plots_dir, exist_ok=True)

# Updated colors for HC and PD
custom_cmap = ListedColormap(['#0075ea', '#eb0077'])

# Method and model order with updated model names
method_order = ['shap', 'mrmr', 'anova']
model_order = ['NoAttention', 'NoTransformer']

# Mapping from index to method and model
for idx in range(6):  # Only 6 plots since we have 2 models and 3 methods
    method_idx = idx // 2
    model_idx = idx % 2  # We have 2 models now
    method = method_order[method_idx]
    model_name = model_order[model_idx]
    ax = axes[idx]

    model_name_in_data = model_name

    if model_name_in_data in umap_data[method]:
        data = umap_data[method][model_name_in_data]
        features = data['features']
        labels = data['labels']

        # Apply UMAP
        reducer = umap.UMAP(random_state=42)
        embedding = reducer.fit_transform(features)

        # Plot
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap=custom_cmap, s=80)
        title_method = method.upper() if method in ['shap', 'anova'] else 'mRMR'
        ax.set_title(f"{title_method} - {model_name}", fontsize=30, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=30, width=2)
    else:
        ax.set_visible(False)

# Add discrete colorbars to the right of each row, aligned with the plots
for i in range(3):
    row_axes = axes[i*2:(i+1)*2]  # 2 per row
    cbar_ax = fig.add_axes([0.92, row_axes[0].get_position().y0, 0.012, row_axes[0].get_position().height])
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=0, vmax=1)), cax=cbar_ax, ticks=[0, 1])
    cbar.set_ticklabels(['HC', 'PD'])
    cbar.ax.tick_params(labelsize=30)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight('bold')

# Save the figure
plot_save_path = os.path.join(output_plots_dir, 'AEno_UMAP.svg')
plt.savefig(plot_save_path)
plt.close(fig)



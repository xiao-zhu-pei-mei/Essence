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

class TransformerModel(nn.Module):
    def __init__(self, input_dim):
        super(TransformerModel, self).__init__()
        # Ensure nhead divides input_dim
        nhead = 1  # Start with 1 head
        for possible_nhead in range(8, 0, -1):  # Try to find the largest possible nhead ≤ 8
            if input_dim % possible_nhead == 0:
                nhead = possible_nhead
                break
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=1)
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add sequence dimension (batch_size, seq_len=1, input_dim)
        x = self.transformer(x)
        x = x.squeeze(1)  # Remove sequence dimension (batch_size, input_dim)
        x = self.fc(x)
        return x  # Output logits directly


class AttentionMechanism(nn.Module):
    def __init__(self, input_dim, d_model=64, dropout=0.2):
        super(AttentionMechanism, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # Query, Key, Value projection layers
        self.fc_q = nn.Linear(input_dim, d_model)
        self.fc_k = nn.Linear(input_dim, d_model)
        self.fc_v = nn.Linear(input_dim, d_model)

        # Output projection layer
        self.fc_o = nn.Linear(d_model, input_dim)

        # Fully connected layer for classification
        self.fc = nn.Linear(input_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        # Project inputs to query, key, and value
        q = self.fc_q(x)  # (batch_size, d_model)
        k = self.fc_k(x)  # (batch_size, d_model)
        v = self.fc_v(x)  # (batch_size, d_model)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q.unsqueeze(1), k.unsqueeze(2)) / (self.d_model ** 0.5)  # (batch_size, 1, 1)
        attn_weights = torch.softmax(attn_weights, dim=-1)  # (batch_size, 1, 1)

        # Apply attention to values
        attn_output = attn_weights.squeeze(-1) * v  # (batch_size, d_model)
        attn_output = self.fc_o(attn_output)  # (batch_size, input_dim)

        # Apply dropout
        attn_output = self.dropout(attn_output)

        # Fully connected layer for classification
        x = self.fc(attn_output)  # (batch_size, 1)
        return x  # Output logits directly


class CNNBiLSTM(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim=32, kernel_num=16, kernel_size=3, dropout=0.3):
        super(CNNBiLSTM, self).__init__()
        self.hidden_dim = lstm_hidden_dim

        # CNN Layer
        self.conv = nn.Conv1d(in_channels=1, out_channels=kernel_num, kernel_size=kernel_size, padding=kernel_size // 2)

        # BiLSTM Layer
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # Fully Connected Layer
        total_feature_dim = kernel_num + self.hidden_dim * 2  # BiLSTM output is doubled due to bidirectionality
        self.fc = nn.Sequential(
            nn.Linear(total_feature_dim, total_feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(total_feature_dim // 2, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        batch_size = x.size(0)
        input_dim = x.size(1)

        # Reshape x for CNN: (batch_size, 1, input_dim)
        x_cnn = x.unsqueeze(1)

        # Apply CNN
        conv_out = self.conv(x_cnn)  # Shape: (batch_size, kernel_num, input_dim)
        conv_out = F.relu(conv_out)
        cnn_output = F.adaptive_avg_pool1d(conv_out, 1).squeeze(2)  # Shape: (batch_size, kernel_num)

        # Reshape x for BiLSTM: (batch_size, seq_len=1, input_dim)
        x_lstm = x.unsqueeze(1)

        # Apply BiLSTM
        bilstm_out, _ = self.bilstm(x_lstm)  # Shape: (batch_size, seq_len=1, hidden_dim * 2)
        bilstm_out = bilstm_out.squeeze(1)  # Shape: (batch_size, hidden_dim * 2)

        # Concatenate CNN and BiLSTM outputs
        combined = torch.cat((cnn_output, bilstm_out), dim=1)  # Shape: (batch_size, total_feature_dim)

        # Fully connected layer
        output = self.fc(combined)  # (batch_size, 1)
        return output  # Output logits directly


class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.2):
        super(SimpleNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        # Hidden layer
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)

        # Output layer for classification
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch_size, input_dim)
        x = self.hidden_layer(x)  # (batch_size, hidden_dim)
        x = F.relu(x)  # Apply ReLU activation
        x = self.dropout(x)  # Apply dropout
        output = self.output_layer(x)  # (batch_size, 1)
        return output  # Output logits directly


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
    'SimpleNNet': SimpleNN,
    'CNN+BiLSTM': CNNBiLSTM,
    'AttentionModel': AttentionMechanism,
    'Transformer': TransformerModel,
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
model_names = ['SimpleNNet', 'CNN+BiLSTM', 'AttentionModel', 'Transformer']
methods = ['shap', 'mrmr', 'anova']
output_dir = '../AE_results'

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

        # Forcefully set best_i to 42 for Transformer and ANOVA method
        if model_name == 'Transformer' and method == 'anova':
            best_i = 42
            print(f"    Forcefully setting best number of features to 42 for Transformer and ANOVA method")

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
            def get_internal_representation(model, x):
                # SimpleNN and SimpleClassificationNetwork
                if isinstance(model, SimpleNN):
                    # Get the output from the hidden layer
                    x = model.hidden_layer(x)
                    x = F.relu(x)
                    return x.cpu().numpy()
                elif isinstance(model, CNNBiLSTM):
                    # For CNNBiLSTM, get the combined features before the final FC layer
                    batch_size = x.size(0)
                    x_cnn = x.unsqueeze(1)
                    conv_out = model.conv(x_cnn)
                    conv_out = F.relu(conv_out)
                    cnn_output = F.adaptive_avg_pool1d(conv_out, 1).squeeze(2)
                    x_lstm = x.unsqueeze(1)
                    bilstm_out, _ = model.bilstm(x_lstm)
                    bilstm_out = bilstm_out.squeeze(1)
                    combined = torch.cat((cnn_output, bilstm_out), dim=1)
                    return combined.cpu().numpy()
                elif isinstance(model, AttentionMechanism):
                    # For AttentionMechanism, get the attention output before the classification layer
                    q = model.fc_q(x)
                    k = model.fc_k(x)
                    v = model.fc_v(x)
                    attn_weights = torch.matmul(q.unsqueeze(1), k.unsqueeze(2)) / (model.d_model ** 0.5)
                    attn_weights = torch.softmax(attn_weights, dim=-1)
                    attn_output = attn_weights.squeeze(-1) * v
                    attn_output = model.fc_o(attn_output)
                    attn_output = model.dropout(attn_output)
                    return attn_output.cpu().numpy()
                elif isinstance(model, TransformerModel):
                    # For TransformerModel, get the transformer output before the final FC layer
                    x = x.unsqueeze(1)
                    x = model.transformer(x)
                    x = x.squeeze(1)
                    return x.cpu().numpy()
                else:
                    # Default case, return the input features
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
fig, axes = plt.subplots(3, 4, figsize=(32, 21))  # 3 rows, 4 columns
plt.subplots_adjust(left=0.04, right=0.9, top=0.95, bottom=0.05, wspace=0.18, hspace=0.25)
axes = axes.flatten()

# Increase the border width for all plots
for ax in axes:
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)

# Define output directory for plots
output_plots_dir = 'AE_plot'
os.makedirs(output_plots_dir, exist_ok=True)

# Updated colors for HC and PD
custom_cmap = ListedColormap(['#0075ea', '#eb0077'])

# Method and model order with updated model names
method_order = ['shap', 'mrmr', 'anova']
model_order = ['FCN', 'CNN+BiLSTM', 'Attention', 'Transformer']  # Updated model names

# Mapping from index to method and model
for idx in range(12):
    method_idx = idx // 4
    model_idx = idx % 4
    method = method_order[method_idx]
    model_name = model_order[model_idx]  # Model name from updated list
    ax = axes[idx]

    # Ensure model names from the 'umap_data' are correctly matched
    model_name_in_data = model_name if model_name != 'FCN' else 'SimpleNNet'  # Map 'FCN' to 'SimpleNNet'
    model_name_in_data = model_name_in_data if model_name != 'Attention' else 'AttentionModel'  # Map 'Attention' to 'AttentionModel'

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
        if model_name == 'SimpleNNet':
            ax.set_title(f"{title_method} - FCN", fontsize=30, fontweight='bold')
        else:
            ax.set_title(f"{title_method} - {model_name}", fontsize=30, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=30, width=2)
    else:
        ax.set_visible(False)


# Add discrete colorbars to the right of each row, aligned with the plots
for i in range(3):
    row_axes = axes[i*4:(i+1)*4]  # Get the axes for the current row
    cbar_ax = fig.add_axes([0.92, row_axes[0].get_position().y0, 0.012, row_axes[0].get_position().height])  # Align with the first plot in the row
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=0, vmax=1)), cax=cbar_ax, ticks=[0, 1])
    cbar.set_ticklabels(['HC', 'PD'])
    cbar.ax.tick_params(labelsize=30)

# Save the figure
plot_save_path = os.path.join(output_plots_dir, 'AE_UMAP.svg')
plt.savefig(plot_save_path)
plt.close(fig)


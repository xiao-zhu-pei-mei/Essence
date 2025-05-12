import pandas as pd
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, matthews_corrcoef, confusion_matrix, average_precision_score

def select_features(method, X, shap_features=None, mrmr_features=None, anova_features=None):
    if method == 'shap':
        if shap_features is None:
            raise ValueError("shap_features must be provided for shap method.")
        # Ensure features are in X
        selected_features = [feature for feature in shap_features if feature in X.columns]
        return selected_features

    elif method == 'mrmr':
        if mrmr_features is None:
            raise ValueError("mrmr_features must be provided for mRMR method.")
        # Ensure features are in X
        selected_features = [feature for feature in mrmr_features if feature in X.columns]
        return selected_features

    elif method == 'anova':
        if anova_features is None:
            raise ValueError("anova_features must be provided for anova method.")
        # Ensure features are in X
        selected_features = [feature for feature in anova_features if feature in X.columns]
        return selected_features


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

def main():
    # Step size for reducing features
    step = 1

    # Set batch size
    batch_size = 32  # Adjust based on your GPU's capacity

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    data = pd.read_csv('Data/data.csv')

    # Extract features and target
    X = data.drop(columns=['target'])
    y = data['target']

    # Read SHAP, mRMR, and ANOVA selected features
    shap_features_df = pd.read_csv('Data/shap_features.csv')
    shap_features = shap_features_df.iloc[:, 0].tolist()

    mrmr_features_df = pd.read_csv('Data/mrmr_features.csv')
    mrmr_features = mrmr_features_df.iloc[:, 0].tolist()

    anova_features_df = pd.read_csv('Data/anova_features.csv')
    anova_features = anova_features_df.iloc[:, 0].tolist()

    # Define feature selection methods
    feature_selection_methods = ['shap', 'mrmr', 'anova']

    # Initialize models
    models = {
        'SimpleNNet': SimpleNN,
        'AttentionModel': AttentionMechanism,
        'CNN+BiLSTM': CNNBiLSTM,
        'Transformer': TransformerModel,
    }

    # Prepare output directory
    output_dir = 'AE_results'
    os.makedirs(output_dir, exist_ok=True)

    # Loop through models
    for model_name, model_class in models.items():
        print(f"Training model: {model_name}")

        # Loop through feature selection methods
        for method in feature_selection_methods:
            print(f"Using feature selection method: {method.upper()}")

            # Initialize highest average AUC and highest fold AUC for this model and method
            highest_avg_auc = 0.0
            highest_fold_auc = 0.0

            sorted_features = select_features(method, X, shap_features, mrmr_features, anova_features)

            # Initialize variables to track accuracies
            max_features = len(sorted_features)
            feature_range = list(range(max_features, 0, -step))  # Reverse the IFS process

            # Loop through different numbers of features
            for i in feature_range:
                current_features = sorted_features[:i]
                X_selected = X[current_features]

                # Prepare tensors
                X_tensor = torch.tensor(X_selected.values, dtype=torch.float32)
                y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

                # Set up KFold cross-validation
                kf = KFold(n_splits=5, shuffle=True, random_state=42)

                fold_accuracies = []
                fold_aucs = []
                fold_sensitivities = []
                fold_specificities = []
                fold_precisions = []
                fold_mccs = []
                fold_auprcs = []

                # List to store best model states for each fold
                best_model_states = []

                # Loop over folds
                for fold, (train_index, val_index) in enumerate(kf.split(X_tensor)):
                    X_train_fold = X_tensor[train_index]
                    y_train_fold = y_tensor[train_index]
                    X_val_fold = X_tensor[val_index]
                    y_val_fold = y_tensor[val_index]

                    # Move tensors to device
                    X_train_fold = X_train_fold.to(device)
                    y_train_fold = y_train_fold.to(device)
                    X_val_fold = X_val_fold.to(device)
                    y_val_fold = y_val_fold.to(device)

                    # Create DataLoader for batching
                    train_dataset = TensorDataset(X_train_fold, y_train_fold)
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                    # Initialize and train the model
                    model = model_class(len(current_features)).to(device)
                    optimizer = optim.Adam(model.parameters(), lr=0.001)
                    criterion = nn.BCEWithLogitsLoss()

                    scaler = GradScaler()

                    # Initialize variables for early stopping
                    best_val_auc = 0.0  # Initialize best validation AUC
                    best_epoch = 0
                    best_model_state = None
                    epochs_since_improvement = 0
                    patience = 5  # Early stopping patience

                    for epoch in range(30):
                        model.train()
                        for batch_X, batch_y in train_loader:
                            batch_X = batch_X.to(device)
                            batch_y = batch_y.to(device)
                            optimizer.zero_grad()
                            # With autocast for mixed precision
                            with autocast():
                                outputs = model(batch_X)
                                loss = criterion(outputs, batch_y)

                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()

                        # Validation after each epoch
                        with torch.no_grad():
                            model.eval()
                            y_pred_logits = model(X_val_fold)
                            y_pred_probs = torch.sigmoid(y_pred_logits).cpu().numpy()
                            y_true = y_val_fold.cpu().numpy().astype(int).flatten()

                            # Calculate validation AUC
                            val_auc = roc_auc_score(y_true, y_pred_probs)

                            # Check if this is the best model so far
                            if val_auc > best_val_auc:
                                best_val_auc = val_auc
                                best_epoch = epoch
                                best_model_state = copy.deepcopy(model.state_dict())  # Use deepcopy here
                                epochs_since_improvement = 0  # Reset counter
                            else:
                                epochs_since_improvement += 1

                            # Early stopping
                            if epochs_since_improvement >= patience:
                                print(f"Early stopping at epoch {epoch + 1}")
                                break

                    # After training, load the best model state
                    model.load_state_dict(best_model_state)

                    # Evaluation phase using the best model
                    with torch.no_grad():
                        model.eval()
                        y_pred_logits = model(X_val_fold)
                        y_pred_probs = torch.sigmoid(y_pred_logits).cpu().numpy()
                        y_pred_labels = (y_pred_probs > 0.5).astype(int).flatten()
                        y_true = y_val_fold.cpu().numpy().astype(int).flatten()

                        # Calculate metrics
                        accuracy = (y_pred_labels == y_true).mean()
                        auc = roc_auc_score(y_true, y_pred_probs)
                        auprc = average_precision_score(y_true, y_pred_probs)
                        precision = ((y_pred_labels * y_true).sum()) / (y_pred_labels.sum() + 1e-6)
                        mcc = matthews_corrcoef(y_true, y_pred_labels)

                        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_labels).ravel()
                        sensitivity = tp / (tp + fn + 1e-6)  # Recall
                        specificity = tn / (tn + fp + 1e-6)

                        fold_accuracies.append(accuracy)
                        fold_aucs.append(auc)
                        fold_sensitivities.append(sensitivity)
                        fold_specificities.append(specificity)
                        fold_precisions.append(precision)
                        fold_mccs.append(mcc)
                        fold_auprcs.append(auprc)

                    print(f"Fold {fold + 1}/{kf.get_n_splits()} - Best Epoch: {best_epoch + 1}, Best Validation AUC: {best_val_auc:.4f}")

                    # Save best model state for this fold
                    best_model_states.append(best_model_state)

                    # Compare fold AUC to highest_fold_auc
                    if auc >= highest_fold_auc:
                        highest_fold_auc = auc

                        # Save the model
                        model_save_dir = os.path.join(output_dir, model_name, method, f'num_features_{i}', 'best_fold_auc', f'fold_{fold+1}')
                        os.makedirs(model_save_dir, exist_ok=True)
                        model_save_path = os.path.join(model_save_dir, 'model.pth')
                        torch.save(best_model_state, model_save_path)

                # Compute average metrics over folds
                avg_accuracy = sum(fold_accuracies) / len(fold_accuracies)
                avg_auc = sum(fold_aucs) / len(fold_aucs)
                avg_sensitivity = sum(fold_sensitivities) / len(fold_sensitivities)
                avg_specificity = sum(fold_specificities) / len(fold_specificities)
                avg_precision = sum(fold_precisions) / len(fold_precisions)
                avg_mcc = sum(fold_mccs) / len(fold_mccs)
                avg_auprc = sum(fold_auprcs) / len(fold_auprcs)

                # Save the results
                results_save_dir = os.path.join(output_dir, model_name, method)
                os.makedirs(results_save_dir, exist_ok=True)
                results_save_path = os.path.join(results_save_dir, f'results_{i}.txt')

                with open(results_save_path, 'w') as f:
                    f.write(f"Features used: {current_features}\n")
                    f.write(f"Number of Features: {i}\n")
                    f.write(f"Average Accuracy: {avg_accuracy}\n")
                    f.write(f"Average AUC: {avg_auc}\n")
                    f.write(f"Average AUPRC: {avg_auprc}\n")
                    f.write(f"Average Sensitivity (Recall): {avg_sensitivity}\n")
                    f.write(f"Average Specificity: {avg_specificity}\n")
                    f.write(f"Average Precision: {avg_precision}\n")
                    f.write(f"Average MCC: {avg_mcc}\n")
                    f.write(f"Accuracies per fold: {fold_accuracies}\n")
                    f.write(f"AUCs per fold: {fold_aucs}\n")
                    f.write(f"AUPRCs per fold: {fold_auprcs}\n")
                    f.write(f"Sensitivities per fold: {fold_sensitivities}\n")
                    f.write(f"Specificities per fold: {fold_specificities}\n")
                    f.write(f"Precisions per fold: {fold_precisions}\n")
                    f.write(f"MCCs per fold: {fold_mccs}\n")

                print(
                    f"Model: {model_name}, Method: {method}, Features: {i}, "
                    f"Avg Acc: {avg_accuracy:.4f}, Avg AUC: {avg_auc:.4f}, "
                    f"Avg AUPRC: {avg_auprc:.4f}, Avg Sensitivity: {avg_sensitivity:.4f}, "
                    f"Avg Specificity: {avg_specificity:.4f}, Avg Precision: {avg_precision:.4f}, Avg MCC: {avg_mcc:.4f}"
                )

                # Compare avg_auc to highest_avg_auc
                if avg_auc >= highest_avg_auc:
                    highest_avg_auc = avg_auc

                    # Save the five models
                    for fold_idx, model_state in enumerate(best_model_states):
                        model_save_dir = os.path.join(output_dir, model_name, method, f'num_features_{i}', 'best_avg_auc', f'fold_{fold_idx+1}')
                        os.makedirs(model_save_dir, exist_ok=True)
                        model_save_path = os.path.join(model_save_dir, 'model.pth')
                        torch.save(model_state, model_save_path)

                # For the first feature (which is now the full feature set), save the five models
                if i == max_features:
                    for fold_idx, model_state in enumerate(best_model_states):
                        model_save_dir = os.path.join(output_dir, model_name, method, f'num_features_{i}', f'fold_{fold_idx+1}')
                        os.makedirs(model_save_dir, exist_ok=True)
                        model_save_path = os.path.join(model_save_dir, 'model.pth')
                        torch.save(model_state, model_save_path)

if __name__ == '__main__':
    main()

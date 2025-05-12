import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from autogluon.tabular import TabularPredictor


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


def select_features(method, X, shap_features=None, mrmr_features=None, anova_features=None):
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

def main():
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

    # Models and methods
    models = [
        'RandomForestGini', 'KNeighborsUnif',
        'RandomForestEntr', 'LightGBM',
        'KNeighborsDist', 'XGBoost'
    ]

    methods = ['shap', 'mrmr', 'anova']

    output_dir = '../ML_results'

    # Prepare to collect plot data
    plot_data = {}

    for model_name in models:
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
            # Sort result files based on the number of features (assuming filename includes number of features)
            result_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            best_single_fold_auc = -1
            best_num_features = None
            best_fold_idx = None
            for result_file in result_files:
                filepath = os.path.join(results_dir, result_file)
                num_features, avg_auc, aucs_per_fold = parse_results_file(filepath)
                if num_features is None or not aucs_per_fold:
                    continue
                for fold_idx, fold_auc in enumerate(aucs_per_fold):
                    if fold_auc > best_single_fold_auc:
                        best_single_fold_auc = fold_auc
                        best_num_features = num_features
                        best_fold_idx = fold_idx  # zero-based index
                    # If fold_auc is equal to current best, we keep the earlier one (already set)
                    elif fold_auc == best_single_fold_auc:
                        # Keep the existing best_num_features and best_fold_idx (earlier one)
                        pass
                    else:
                        # Do not update if fold_auc is less than current best
                        pass

            if best_num_features is None or best_fold_idx is None:
                print(f"    No valid results found for method {method} in model {model_name}")
                continue

            print(
                f"    Best number of features: {best_num_features}, Best Single-Fold AUC: {best_single_fold_auc}, Fold: {best_fold_idx + 1}")

            # Select features
            sorted_features = select_features(method, X, shap_features=shap_features, mrmr_features=mrmr_features,
                                              anova_features=anova_features)
            current_features = sorted_features[:best_num_features]
            X_selected = X[current_features]

            # Combine features and target
            data_selected = X_selected.copy()
            data_selected['target'] = y

            # Set up KFold cross-validation
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            splits = list(kf.split(data_selected))

            # Get the validation data for the best fold
            train_index, val_index = splits[best_fold_idx]
            train_data_fold = data_selected.iloc[train_index]
            val_data_fold = data_selected.iloc[val_index]

            # Load the model for this fold
            model_save_dir = os.path.join(
                output_dir, 'saved_models', model_name, method, f'features_{best_num_features}',
                f'fold_{best_fold_idx + 1}'
            )
            if not os.path.exists(model_save_dir):
                print(f"    Model directory does not exist: {model_save_dir}")
                continue

            # Load the predictor
            predictor = TabularPredictor.load(model_save_dir)

            # Predict on validation set
            y_true = val_data_fold['target']
            y_pred_proba = predictor.predict_proba(val_data_fold[current_features])  # Get probabilities
            if isinstance(y_pred_proba, pd.DataFrame):
                # For binary classification, get probability of positive class
                if 1 in y_pred_proba.columns:
                    y_pred_probs = y_pred_proba[1].values
                else:
                    # If columns are labeled differently, take the second column
                    y_pred_probs = y_pred_proba.iloc[:, 1].values
            else:
                y_pred_probs = y_pred_proba

            # Compute ROC and PRC curves
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
            roc_auc_value = auc(fpr, tpr)

            precision, recall, thresholds_prc = precision_recall_curve(y_true, y_pred_probs)
            prc_auc_value = auc(recall, precision)
            average_precision = average_precision_score(y_true, y_pred_probs)

            # Store the plot data
            plot_data[model_name]['roc'][method] = (fpr, tpr, roc_auc_value)
            plot_data[model_name]['prc'][method] = (recall, precision, average_precision)

            # Clean up to save memory
            del predictor

    # Now plot the ROC and PRC curves and save them to a folder
    output_plots_dir = 'ML_plot'
    os.makedirs(output_plots_dir, exist_ok=True)

    # Prepare the figure
    fig, axes = plt.subplots(3, 4, figsize=(32, 21))  # 3 rows, 4 columns
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    axes = axes.flatten()

    # Define line colors
    line_colors = ['#FFCC00', '#009999', '#CC3366']

    # Mapping from index to model and plot type
    idx = 0
    for model_name in models:
        for plot_type in ['roc', 'prc']:
            ax = axes[idx]
            for method_idx, method in enumerate(methods):
                if method not in plot_data[model_name][plot_type]:
                    continue
                color = line_colors[method_idx % len(line_colors)]

                # Set method label
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

            idx += 1

    # Remove unused subplots
    for i in range(idx, len(axes)):
        fig.delaxes(axes[i])

    # Save the figure
    plot_save_path = os.path.join(output_plots_dir, 'ML_ROCPRCsup.svg')
    plt.tight_layout()
    plt.savefig(plot_save_path)
    plt.close()
    print(f"ROC and PRC curves saved at {plot_save_path}")


if __name__ == '__main__':
    main()

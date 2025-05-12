import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
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

    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X, y, test_size=0.2, random_state=42)

    models = [
        'LightGBMXT', 'ExtraTreesGini', 'ExtraTreesEntr', 'CatBoost'
    ]
    methods = ['shap', 'mrmr', 'anova']

    output_dir = '../ML_results'
    output_plots_dir = 'ML_plot'
    os.makedirs(output_plots_dir, exist_ok=True)

    num_rows = 3
    num_cols = 4
    fontsize = 30

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(32, num_rows * 7))
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    axes = axes.flatten()
    plot_idx = 0

    for method in methods:
        print(f"Processing feature selection method: {method.upper()}")
        for model_name in models:
            print(f"  Processing model: {model_name}")
            results_dir = os.path.join(output_dir, model_name, method)
            if not os.path.exists(results_dir):
                print(f"    Results directory does not exist: {results_dir}")
                continue

            result_files = [f for f in os.listdir(results_dir) if f.startswith('results_') and f.endswith('.txt')]
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
                        best_fold_idx = fold_idx

            if best_num_features is None or best_fold_idx is None:
                print(f"    No valid results found for model {model_name} with method {method}")
                continue

            print(f"    Best number of features: {best_num_features}, Best Single-Fold AUC: {best_single_fold_auc}, Fold: {best_fold_idx+1}")

            sorted_features = select_features(method, X, shap_features=shap_features, mrmr_features=mrmr_features, anova_features=anova_features)
            current_features = sorted_features[:best_num_features]

            X_test_selected = X_test_full[current_features]
            y_test = y_test_full

            model_save_dir = os.path.join(
                output_dir, 'saved_models', model_name, method, f'features_{best_num_features}', f'fold_{best_fold_idx+1}'
            )
            if not os.path.exists(model_save_dir):
                print(f"    Model directory does not exist: {model_save_dir}")
                continue

            predictor = TabularPredictor.load(model_save_dir)
            y_pred = predictor.predict(X_test_selected)
            y_pred_proba = predictor.predict_proba(X_test_selected)
            y_pred_probs = y_pred_proba[1].values if isinstance(y_pred_proba, pd.DataFrame) and 1 in y_pred_proba.columns else y_pred_proba.iloc[:, 1].values if isinstance(y_pred_proba, pd.DataFrame) else y_pred_proba
            test_auc = roc_auc_score(y_test, y_pred_probs)
            print(f"    Test AUC: {test_auc:.4f}")

            X_test_pos = X_test_selected[y_pred == 1]
            X_test_neg = X_test_selected[y_pred == 0]
            if X_test_pos.empty or X_test_neg.empty:
                print(f"    Insufficient data in one of the predicted classes for model {model_name} with method {method}")
                continue

            mean_test_pos = X_test_pos.mean()
            std_test_pos = X_test_pos.std()
            mean_test_neg = X_test_neg.mean()
            std_test_neg = X_test_neg.std()

            features = range(1, len(mean_test_pos) + 1)
            ax = axes[plot_idx]
            dotsize = 60
            ax.scatter(features, mean_test_pos, color='#eb0077', marker='o', label='PD-AVG', s=dotsize)
            ax.scatter(features, std_test_pos, color='#eb0077', marker='+', label='PD-SD', s=dotsize)
            ax.scatter(features, mean_test_neg, color='#0075ea', marker='o', label='HC-AVG', s=dotsize)
            ax.scatter(features, std_test_neg, color='#0075ea', marker='+', label='HC-SD', s=dotsize)

            ax.set_ylabel(' ', fontsize=fontsize, weight='bold')
            ax.tick_params(axis='both', which='major', labelsize=30)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight('bold')

            ax.set_ylim(-0.5, 3)
            ax.spines['right'].set_linewidth(2)
            ax.spines['left'].set_linewidth(2)
            ax.spines['bottom'].set_linewidth(2)
            ax.spines['top'].set_linewidth(2)

            method_title = 'mRMR' if method == 'mrmr' else method.upper()
            ax.set_title(f'{method_title} - {model_name}', fontsize=fontsize, weight='bold')

            legend = ax.legend(loc='upper right', fontsize=30, markerscale=1.8, frameon=False)
            for text in legend.get_texts():
                text.set_fontweight('bold')

            plot_idx += 1
            del predictor

    plt.tight_layout(pad=3, w_pad=3, h_pad=3)
    plot_save_path = os.path.join(output_plots_dir, 'ML_SD.svg')
    plt.savefig(plot_save_path)
    plt.close()
    print(f"Feature statistics plots saved at {plot_save_path}")

if __name__ == '__main__':
    main()

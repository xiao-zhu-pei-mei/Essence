import os
import shutil
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    average_precision_score,
)
from sklearn.model_selection import KFold
from autogluon.tabular import TabularPredictor

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

def main():
    # Load data
    data = pd.read_csv('Data/data.csv')

    # Extract features and target
    X = data.drop(columns=['target'])
    y = data['target']

    # Read feature selection results
    shap_features_df = pd.read_csv('Data/shap_features.csv')
    shap_features = shap_features_df.iloc[:, 0].tolist()

    mrmr_features_df = pd.read_csv('Data/mrmr_features.csv')
    mrmr_features = mrmr_features_df.iloc[:, 0].tolist()

    anova_features_df = pd.read_csv('Data/anova_features.csv')
    anova_features = anova_features_df.iloc[:, 0].tolist()

    # Define feature selection methods
    feature_selection_methods = ['shap', 'mrmr', 'anova']

    # Set step size for adding features
    step = 1  # Adjust as needed

    # Define models to use
    models = {
        'LightGBM': {'GBM': {'ag_args_fit': {'num_workers': 8}}},
        'LightGBMXT': {'GBM': {'extra_trees': True, 'ag_args_fit': {'num_workers': 8}}},
        'CatBoost': {'CAT': {'ag_args_fit': {'num_workers': 8}}},
        'XGBoost': {'XGB': {'ag_args_fit': {'num_workers': 8}}},
        'RandomForestGini': {'RF': {'criterion': 'gini', 'ag_args_fit': {'num_workers': 8}}},
        'RandomForestEntr': {'RF': {'criterion': 'entropy', 'ag_args_fit': {'num_workers': 8}}},
        'ExtraTreesGini': {'XT': {'criterion': 'gini', 'ag_args_fit': {'num_workers': 8}}},
        'ExtraTreesEntr': {'XT': {'criterion': 'entropy', 'ag_args_fit': {'num_workers': 8}}},
        'KNeighborsUnif': {'KNN': {'weights': 'uniform', 'ag_args_fit': {'num_workers': 8}}},
        'KNeighborsDist': {'KNN': {'weights': 'distance', 'ag_args_fit': {'num_workers': 8}}},
    }

    output_dir = 'ML_results'
    os.makedirs(output_dir, exist_ok=True)

    for model_name, hyperparameters in models.items():
        print(f"Processing model: {model_name}")
        model_output_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)

        for method_idx, method in enumerate(feature_selection_methods):
            print(f"Processing feature selection method: {method}")
            method_output_dir = os.path.join(model_output_dir, method)
            os.makedirs(method_output_dir, exist_ok=True)

            # Select features
            if method == 'shap':
                sorted_features = select_features(method, X, shap_features=shap_features)
            elif method == 'mrmr':
                sorted_features = select_features(method, X, mrmr_features=mrmr_features)
            elif method == 'anova':
                sorted_features = select_features(method, X, anova_features=anova_features)

            # Initialize variables to keep track of highest AUCs
            highest_avg_auc = 0.0
            highest_fold_auc = 0.0

            max_features = len(sorted_features)
            feature_range = list(range(step, max_features + 1, step))

            # Loop over different numbers of features
            for i in feature_range:
                print(f"Processing {i} features")
                current_features = sorted_features[:i]
                X_selected = X[current_features]

                # Combine features and target
                data_selected = X_selected.copy()
                data_selected['target'] = y

                # Set up KFold cross-validation
                kf = KFold(n_splits=5, shuffle=True, random_state=42)

                fold_accuracies = []
                fold_aucs = []
                fold_sensitivities = []
                fold_specificities = []
                fold_precisions = []
                fold_mccs = []
                fold_auprcs = []

                fold_models_saved = [False] * 5  # Keep track of whether models have been saved
                for fold, (train_index, val_index) in enumerate(kf.split(data_selected)):
                    print(f"Fold {fold + 1}")
                    train_data_fold = data_selected.iloc[train_index]
                    val_data_fold = data_selected.iloc[val_index]

                    # Define temporary model directory
                    temp_model_dir = os.path.join(
                        output_dir, 'temp_models', model_name, method, f'features_{i}', f'fold_{fold + 1}'
                    )
                    os.makedirs(temp_model_dir, exist_ok=True)

                    # Train the model using AutoGluon
                    predictor = TabularPredictor(
                        label='target',
                        eval_metric='roc_auc',
                        verbosity=0,
                        path=temp_model_dir
                    ).fit(
                        train_data=train_data_fold,
                        time_limit=300,
                        hyperparameters=hyperparameters,
                        presets='medium_quality',
                    )

                    # Predict on validation set
                    y_true = val_data_fold['target']
                    y_pred = predictor.predict(val_data_fold)
                    y_pred_proba = predictor.predict_proba(val_data_fold)  # Get probabilities

                    # Calculate metrics
                    accuracy = accuracy_score(y_true, y_pred)
                    auc = roc_auc_score(y_true, y_pred_proba.iloc[:, 1])
                    auprc = average_precision_score(y_true, y_pred_proba.iloc[:, 1])
                    precision = ((y_pred & y_true).sum()) / (y_pred.sum() + 1e-6)
                    mcc = matthews_corrcoef(y_true, y_pred)

                    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                    sensitivity = tp / (tp + fn + 1e-6)  # Recall
                    specificity = tn / (tn + fp + 1e-6)

                    fold_accuracies.append(accuracy)
                    fold_aucs.append(auc)
                    fold_sensitivities.append(sensitivity)
                    fold_specificities.append(specificity)
                    fold_precisions.append(precision)
                    fold_mccs.append(mcc)
                    fold_auprcs.append(auprc)

                    # Model saving logic
                    fold_model_saved = False
                    if i == step:
                        # Save models for the first feature
                        permanent_model_dir = os.path.join(
                            output_dir, 'saved_models', model_name, method, f'features_{i}', f'fold_{fold + 1}'
                        )
                        shutil.move(temp_model_dir, permanent_model_dir)
                        fold_model_saved = True
                        print(f"Saved model for fold {fold + 1} with {i} features (first feature set).")
                    else:
                        if auc > highest_fold_auc:
                            highest_fold_auc = auc
                            permanent_model_dir = os.path.join(
                                output_dir, 'saved_models', model_name, method, f'features_{i}', f'fold_{fold + 1}'
                            )
                            shutil.move(temp_model_dir, permanent_model_dir)
                            fold_model_saved = True
                            print(
                                f"Saved model for fold {fold + 1} with {i} features (new highest fold AUC: {auc:.4f})."
                            )
                        else:
                            # Delete the temporary model directory
                            shutil.rmtree(temp_model_dir)
                            print(f"Deleted model for fold {fold + 1} with {i} features (AUC: {auc:.4f}).")

                    fold_models_saved[fold] = fold_model_saved

                    # Clean up to save memory
                    del predictor

                # Compute average metrics over folds
                avg_accuracy = np.mean(fold_accuracies)
                avg_auc = np.mean(fold_aucs)
                avg_sensitivity = np.mean(fold_sensitivities)
                avg_specificity = np.mean(fold_specificities)
                avg_precision = np.mean(fold_precisions)
                avg_mcc = np.mean(fold_mccs)
                avg_auprc = np.mean(fold_auprcs)

                # Save the results
                results_save_dir = method_output_dir
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

                # Average AUC saving logic
                if i == step:
                    highest_avg_auc = avg_auc
                    print(f"Set highest average AUC to {highest_avg_auc:.4f} (first feature set).")
                else:
                    if avg_auc > highest_avg_auc:
                        highest_avg_auc = avg_auc
                        # Move any unsaved models to permanent directory
                        for fold in range(5):
                            if not fold_models_saved[fold]:
                                temp_model_dir = os.path.join(
                                    output_dir, 'temp_models', model_name, method, f'features_{i}', f'fold_{fold + 1}'
                                )
                                permanent_model_dir = os.path.join(
                                    output_dir, 'saved_models', model_name, method, f'features_{i}', f'fold_{fold + 1}'
                                )
                                if os.path.exists(temp_model_dir):
                                    shutil.move(temp_model_dir, permanent_model_dir)
                                    print(
                                        f"Saved model for fold {fold + 1} with {i} features (new highest average AUC)."
                                    )
                        print(f"Updated highest average AUC to {highest_avg_auc:.4f}.")
                    else:
                        # Delete any unsaved models
                        for fold in range(5):
                            if not fold_models_saved[fold]:
                                temp_model_dir = os.path.join(
                                    output_dir, 'temp_models', model_name, method, f'features_{i}', f'fold_{fold + 1}'
                                )
                                if os.path.exists(temp_model_dir):
                                    shutil.rmtree(temp_model_dir)
                                    print(f"Deleted model for fold {fold + 1} with {i} features (average AUC not improved).")

                print(
                    f"Model: {model_name}, Method: {method}, Features: {i}, "
                    f"Avg Acc: {avg_accuracy:.4f}, Avg AUC: {avg_auc:.4f}, "
                    f"Avg AUPRC: {avg_auprc:.4f}, Avg Sensitivity: {avg_sensitivity:.4f}, "
                    f"Avg Specificity: {avg_specificity:.4f}, Avg Precision: {avg_precision:.4f}, Avg MCC: {avg_mcc:.4f}"
                )

    # Clean up temporary directories
    temp_models_dir = os.path.join(output_dir, 'temp_models')
    if os.path.exists(temp_models_dir):
        shutil.rmtree(temp_models_dir)
        print("Cleaned up temporary model directories.")

if __name__ == '__main__':
    main()

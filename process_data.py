import pandas as pd
import numpy as np
import os
import shap
import pymrmr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_classif

# ========== Step 1: 数据预处理 ==========
def load_and_preprocess_data():
    df1 = pd.read_csv('Data/2020-11-20_CSF_HBS_DIA_Proteins_Percentile025.csv')
    df2 = pd.read_csv('Data/2020-11-20_CSF_LCC_DIA_Proteins_Percentile025.csv')
    df_HBS = pd.read_excel('Data/mmc2.xlsx', sheet_name='HBS')
    df_LCC = pd.read_excel('Data/mmc2.xlsx', sheet_name='LCC')

    HBS_ID = df_HBS.iloc[:, 0].astype(str)
    HBS_TG = df_HBS.iloc[:, 2]
    HBS_DP = df_HBS.iloc[:, 17]
    LCC_ID = df_LCC.iloc[:, 0].astype(str)
    LCC_TG = df_LCC.iloc[:, 2]
    LCC_DP = df_LCC.iloc[:, 12]

    HBS_nan_IDs = HBS_ID[HBS_DP.isna()]
    LCC_nan_IDs = LCC_ID[LCC_DP.isna()]

    def get_id_from_column(col_name):
        if '[' in col_name and ']' in col_name:
            return col_name.split('[')[1].split(']')[0]
        return None

    df1_columns_map = {get_id_from_column(col): col for col in df1.columns[-96:] if get_id_from_column(col)}
    df2_columns_map = {get_id_from_column(col): col for col in df2.columns[-130:] if get_id_from_column(col)}

    HBS_columns = [df1_columns_map[id_] for id_ in HBS_nan_IDs if id_ in df1_columns_map]
    LCC_columns = [df2_columns_map[id_] for id_ in LCC_nan_IDs if id_ in df2_columns_map]

    df1_selected = df1[HBS_columns]
    df2_selected = df2[LCC_columns]

    HBS_target = [1 if tg == 'PD' else 0 for tg in HBS_TG[HBS_DP.isna()]]
    LCC_target = [1 if tg == 'PD' else 0 for tg in LCC_TG[LCC_DP.isna()]]
    target = HBS_target + LCC_target

    df1['combined'] = df1.iloc[:, 3].fillna('') + '/' + df1.iloc[:, 4].fillna('')
    df2['combined'] = df2.iloc[:, 3].fillna('') + '/' + df2.iloc[:, 4].fillna('')

    df1_selected.index = df1['combined']
    df2_selected.index = df2['combined']
    common_proteins = df1_selected.index.intersection(df2_selected.index)

    data_df1 = df1_selected.loc[common_proteins]
    data_df2 = df2_selected.loc[common_proteins]
    data = pd.concat([data_df1, data_df2], axis=1).T

    data = (data - data.mean()) / data.std()
    data['target'] = target
    os.makedirs('Data', exist_ok=True)
    data.to_csv('Data/data.csv', index=True)
    return data

# ========== Step 2: 加载数据 ==========
data = load_and_preprocess_data()
data = pd.read_csv('Data/data.csv')
data = data.select_dtypes(include=[np.number])
y = data['target']
X = data.drop(columns=['target'])

# ========== Step 3: ANOVA ==========
F_values, _ = f_classif(X, y)
anova_scores = pd.DataFrame({'Feature': X.columns, 'ANOVA_Score': F_values})
anova_scores.sort_values(by='ANOVA_Score', ascending=False).to_csv('Data/anova_features.csv', index=False)
print("✓ ANOVA 特征评分已保存：Data/anova_features.csv")

# ========== Step 4: mRMR ==========
mrmr_features = pymrmr.mRMR(data, 'MIQ', data.shape[1])
correlations = data[mrmr_features].apply(lambda col: col.corr(y))
mrmr_df = pd.DataFrame({'Feature': mrmr_features, 'Score': correlations})
mrmr_df = mrmr_df[mrmr_df['Feature'] != 'target']
mrmr_df.to_csv('Data/mrmr_features.csv', index=False)
print("✓ mRMR 特征评分已保存（已去除 target）：Data/mrmr_features.csv")


# ========== Step 5: SHAP ==========
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_train)

shap_class = shap_values[1] if isinstance(shap_values, list) else shap_values
mean_abs_shap = np.abs(shap_class).mean(axis=0)

# 处理异常维度
if mean_abs_shap.ndim == 2 and mean_abs_shap.shape[1] == 2:
    mean_abs_shap = mean_abs_shap[:, 1]
elif mean_abs_shap.ndim == 1 and mean_abs_shap.shape[0] == 2 * X_train.shape[1]:
    mean_abs_shap = mean_abs_shap[X_train.shape[1]:]
mean_abs_shap = mean_abs_shap.ravel()

shap_df = pd.DataFrame({'Feature': X_train.columns, 'Score': mean_abs_shap})
shap_df.sort_values(by='Score', ascending=False).to_csv('Data/shap_features.csv', index=False)
print("✓ SHAP 特征评分已保存：Data/shap_features.csv")

import os
import re
import matplotlib.pyplot as plt
import ast

def plot_autogluon_ifs_graph_auc():
    # ========== 1) 在这里配置手动标注位置 (model_name -> method -> label_position) ==========
    # label_position 编码：
    #   1 = 右上角
    #   2 = 左上角
    #   3 = 左下角
    #   4 = 右下角
    #   5 = 中上
    #   6 = 中下
    #
    # 如果没有配置，则默认使用 1（右上角）。
    manual_label_positions = {
        "RandomForestGini": {"shap": 1,"mrmr": 1,"anova": 1,},
        "KNeighborsUnif": {"shap": 1,"mrmr": 1,"anova": 1},
        "RandomForestEntr": {"shap": 1,"mrmr": 1,"anova": 1},
        "LightGBM": {"shap": 1,"mrmr": 1,"anova": 1},
        "KNeighborsDist": {"shap": 1,"mrmr": 1,"anova": 1},
        "XGBoost": {"shap": 1,"mrmr": 1,"anova": 1},
    }
    # ==================================================================================

    def get_label_offset(pos):
        if pos == 1:    # 右上
            return (5, 5, 'left', 'bottom')
        elif pos == 2:  # 左上
            return (-5, 5, 'right', 'bottom')
        elif pos == 3:  # 左下
            return (-5, -5, 'right', 'top')
        elif pos == 4:  # 右下
            return (5, -5, 'left', 'top')
        elif pos == 5:  # 中上
            return (0, 5, 'center', 'bottom')
        elif pos == 6:  # 中下
            return (0, -5, 'center', 'top')
        else:
            # 万一输错，就用右上角
            return (5, 5, 'left', 'bottom')

    results_dir = '../ML_results'
    output_dir = 'ML_plot'
    os.makedirs(output_dir, exist_ok=True)

    # 定义模型和特征选择方法
    models = [
        'RandomForestGini', 'KNeighborsUnif',
        'RandomForestEntr', 'LightGBM',
        'KNeighborsDist', 'XGBoost'
    ]
    feature_selection_methods = ['shap', 'mrmr', 'anova']

    # 颜色和线型
    colors = ['#FFCC00', '#009999', '#CC3366']
    linestyles = ['-', '-', '-']

    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(24, 14), sharey=False)
    plt.subplots_adjust(wspace=0.18)

    # 方法标签（legend 中仅保留方法名称）
    method_labels = {
        'shap': 'SHAP',
        'mrmr': 'mRMR',
        'anova': 'ANOVA'
    }

    # 字体大小（并将所有文字加粗）
    label_fontsize = 30
    tick_fontsize = 24
    legend_fontsize = 22
    annotation_fontsize = 22

    for model_idx, model_name in enumerate(models):
        ax = axes[model_idx // 3, model_idx % 3]
        max_features = 0  # 用来设置 x 轴范围

        for method_idx, method in enumerate(feature_selection_methods):
            method_results_dir = os.path.join(results_dir, model_name, method)
            if not os.path.exists(method_results_dir):
                continue

            feature_counts = [0]  # 从 0 开始，为了曲线更好看（起始点）
            aucs = [0.4]          # 最低连到 0.4

            # 读取 .txt 结果文件
            result_files = [f for f in os.listdir(method_results_dir) if f.endswith('.txt')]
            # 按特征数排序
            result_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))

            for filename in result_files:
                if filename.endswith('.txt'):
                    num_features = int(re.findall(r'\d+', filename)[0])
                    if num_features > max_features:
                        max_features = num_features

                    file_path = os.path.join(method_results_dir, filename)
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                        max_auc = None
                        for line in lines:
                            if 'AUCs per fold' in line:
                                aucs_list_str = line.strip().split(':', 1)[-1].strip()
                                aucs_list = ast.literal_eval(aucs_list_str)
                                max_auc = max(aucs_list)  # 可换成平均值 np.mean(aucs_list) 等
                                break
                        if max_auc is not None:
                            feature_counts.append(num_features)
                            aucs.append(max_auc)

            if len(feature_counts) == 1:
                # 如果只有起始点，没有任何真实数据，跳过
                continue

            # 找到最高 AUC
            highest_auc_overall = max(aucs)
            best_num_features = feature_counts[aucs.index(highest_auc_overall)]

            # 绘制曲线
            label = method_labels[method]
            ax.plot(
                feature_counts,
                aucs,
                label=label,
                color=colors[method_idx % len(colors)],
                linestyle=linestyles[method_idx % len(linestyles)],
                linewidth=2
            )
            # 标记最高点
            ax.scatter(best_num_features, highest_auc_overall, color='green', alpha=0.5)
            ax.hlines(highest_auc_overall, -40, best_num_features, colors='gray', linestyles='--')
            ax.vlines(best_num_features, 0.4, highest_auc_overall, colors='gray', linestyles='--')

            # 读取手动配置的标注位置编号
            label_pos = manual_label_positions.get(model_name, {}).get(method, 1)
            offset_x, offset_y, ha, va = get_label_offset(label_pos)

            # 添加文本注释 (要使用 annotate 才能用 xytext/textcoords)
            ax.annotate(
                f'({best_num_features}, {highest_auc_overall:.4f})',
                xy=(best_num_features, highest_auc_overall),
                xytext=(offset_x, offset_y),
                textcoords='offset points',
                ha=ha,
                va=va,
                fontsize=annotation_fontsize,
                fontweight='bold'  # 标注加粗
            )

        if max_features == 0:
            continue

        # 调整坐标轴范围
        ax.set_xlim(-40, max_features + 10)
        ax.set_ylim(0.4, 1.01)

        # 设置轴标签、标题 (加粗)
        ax.set_xlabel('Number of Features', fontsize=label_fontsize, fontweight='bold', fontstyle='italic')
        ax.set_ylabel('AUC', fontsize=label_fontsize, fontweight='bold', fontstyle='italic')
        ax.set_title(f'{model_name}', fontsize=label_fontsize, fontweight='bold')

        # 坐标刻度加粗
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        # 强行设置刻度文字加粗
        for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
            tick_label.set_fontweight('bold')

        # 美化边框
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)

        # 图例并加粗
        legend = ax.legend(fontsize=legend_fontsize, loc='lower right', frameon=False)
        # 调整图例文字加粗
        for text in legend.get_texts():
            text.set_fontweight('bold')
        # 加粗图例中曲线
        for line in legend.get_lines():
            line.set_linewidth(8)

    plt.tight_layout()
    plt.subplots_adjust(right=0.98)

    # 保存图像
    plot_filename = os.path.join(output_dir, 'ML_IFSsup.svg')
    plt.savefig(plot_filename)
    plt.close()
    print(f"Combined plot saved at {plot_filename}")

if __name__ == '__main__':
    plot_autogluon_ifs_graph_auc()

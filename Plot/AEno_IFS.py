import os
import re
import matplotlib.pyplot as plt
import ast  # 用于安全解析字符串为列表

def plot_combined_ifs_graph_auc():
    # ================= Manual annotation position =================
    # 1 = 右上, 2 = 左上, 3 = 左下, 4 = 右下, 5 = 中上, 6 = 中下
    manual_label_positions = {
        'NoAttention': {'shap': 1, 'mrmr': 1, 'anova': 1},
        'NoTransformer': {'shap': 1, 'mrmr': 1, 'anova': 1},
    }

    def get_label_offset(pos):
        if pos == 1:
            return (5, 5, 'left', 'bottom')
        elif pos == 2:
            return (-5, 5, 'right', 'bottom')
        elif pos == 3:
            return (-5, -5, 'right', 'top')
        elif pos == 4:
            return (5, -5, 'left', 'top')
        elif pos == 5:
            return (0, 5, 'center', 'bottom')
        elif pos == 6:
            return (0, -5, 'center', 'top')
        else:
            return (5, 5, 'left', 'bottom')

    results_dir = '../AEno_results'
    output_dir = 'AEno_plot'
    os.makedirs(output_dir, exist_ok=True)

    models = ['NoAttention', 'NoTransformer']
    feature_selection_methods = ['shap', 'mrmr', 'anova']

    colors = ['#FFCC00', '#009999', '#CC3366']
    linestyles = ['-', '-', '-']

    fig, axes = plt.subplots(1, len(models), figsize=(16, 7), sharey=False)
    plt.subplots_adjust(wspace=0.18)

    for model_idx, model_name in enumerate(models):
        ax = axes[model_idx]
        max_features = 0
        min_features = float('inf')

        for method_idx, method in enumerate(feature_selection_methods):
            method_results_dir = os.path.join(results_dir, model_name, method)
            if not os.path.exists(method_results_dir):
                continue

            feature_counts = [0]
            aucs = [0.4]

            result_files = [f for f in os.listdir(method_results_dir) if f.endswith('.txt')]
            result_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))

            for filename in result_files:
                num_features = int(re.findall(r'\d+', filename)[0])
                max_features = max(max_features, num_features)
                min_features = min(min_features, num_features)

                file_path = os.path.join(method_results_dir, filename)
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    max_auc = None
                    for line in lines:
                        if 'AUCs per fold' in line:
                            aucs_list_str = line.strip().split(':', 1)[-1].strip()
                            aucs_list = ast.literal_eval(aucs_list_str)
                            max_auc = max(aucs_list)
                            break
                    if max_auc is not None:
                        feature_counts.append(num_features)
                        aucs.append(max_auc)

            if len(feature_counts) == 0:
                continue

            highest_auc_overall = max(aucs)
            indices_with_highest_auc = [i for i, auc in enumerate(aucs) if auc == highest_auc_overall]
            best_num_features = min([feature_counts[i] for i in indices_with_highest_auc])

            method_label = 'mRMR' if method == 'mrmr' else method.upper()
            label = f'{method_label}'

            ax.plot(
                feature_counts,
                aucs,
                label=label,
                color=colors[method_idx % len(colors)],
                linestyle=linestyles[method_idx % len(linestyles)],
                linewidth=2
            )
            ax.scatter(best_num_features, highest_auc_overall, color='green', alpha=0.5)
            ax.hlines(highest_auc_overall, -40, best_num_features, colors='gray', linestyles='--')
            ax.vlines(best_num_features, 0.4, highest_auc_overall, colors='gray', linestyles='--')

            label_pos = manual_label_positions.get(model_name, {}).get(method, 1)
            offset_x, offset_y, ha, va = get_label_offset(label_pos)

            ax.annotate(
                f'({best_num_features}, {highest_auc_overall:.4f})',
                xy=(best_num_features, highest_auc_overall),
                xytext=(offset_x, offset_y),
                textcoords='offset points',
                ha=ha,
                va=va,
                fontsize=22,
                fontweight='bold'
            )

        if max_features == 0:
            continue

        ax.set_xlim(-40, max_features + 20)
        ax.set_ylim(0.4, 1.01)

        ax.set_xlabel('Number of Features', fontsize=30, fontweight='bold', fontstyle='italic')
        ax.set_ylabel('AUC', fontsize=30, fontweight='bold', fontstyle='italic')
        ax.set_title(model_name, fontsize=30, fontweight='bold')

        ax.tick_params(axis='both', which='major', labelsize=30)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight('bold')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)

        legend = ax.legend(fontsize=22, loc='lower right', frameon=False, bbox_to_anchor=(1.02, 0))
        for text in legend.get_texts():
            text.set_fontweight('bold')
        for line in legend.get_lines():
            line.set_linewidth(8)

    plt.tight_layout()
    plt.subplots_adjust(right=0.98)
    plot_filename = os.path.join(output_dir, 'AEno_IFS.svg')
    plt.savefig(plot_filename)
    plt.close()
    print(f"Combined plot saved at {plot_filename}")

if __name__ == '__main__':
    plot_combined_ifs_graph_auc()

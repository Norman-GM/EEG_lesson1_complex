import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.multiclass import unique_labels
import os
def draw_confusion_matrix(model, classes_dict, sub, dataloader, save_path, normalize=True):
    """
    Confusion matrix visualization
    :return:
    """
    model.eval()
    model_name = model.__class__.__name__
    with torch.no_grad():
        all_preds = []
        all_labels = []
        for data, target in dataloader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            output = model(data)
            _, preds = torch.max(output, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 获取类别名称
    if classes_dict is not None:
        classes = [classes_dict[i] for i in sorted(classes_dict.keys())]
    else:
        classes = unique_labels(y_true, y_pred)

    # 归一化处理
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        vmin, vmax = 0, 1
    else:
        fmt = 'd'
        vmin, vmax = None, None

    # 创建DataFrame
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)

    # 创建图形
    plt.figure(figsize=(8, 6))
    # 使用自定义颜色渐变
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    sns.set(font_scale=1.2)  # 增大字体
    plt.style.use('seaborn-whitegrid')  # 使用更美观的样式
    # 使用seaborn绘制热力图
    ax = sns.heatmap(cm_df, annot=True, fmt=fmt, cmap=cmap,
                     vmin=vmin, vmax=vmax, cbar=True,
                    linewidths=0.5, linecolor='gray')

    # 设置标题和标签
    ax.set_title(f'Confusion Matrix of {model_name}, sub: {sub}', fontsize=16)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)

    # 调整颜色条
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    # 在绘制热力图后添加对角线高亮
    for i in range(len(classes)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='red', lw=1))
    # 调整布局
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
    else:
        plt.show()

    plt.close()

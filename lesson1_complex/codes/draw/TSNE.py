import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch
import os
def tsne(model, layer_name,classes_dict,sub, dataloader, save_path):
    """
    t-SNE visualization
    :return:
    """


    # 标准化数据
    # 注册hook获取中间层输出
    activation = {}
    features = []
    labels = []
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    # 获取指定层并注册hook
    layer = dict([*model.named_modules()])[layer_name]
    layer.register_forward_hook(get_activation(layer_name))
    # 获取数据
    model.eval()
    model_name  = model.__class__.__name__
    with torch.no_grad():
        for data, target in dataloader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            _ = model(data)
            features.append(activation[layer_name].cpu().numpy())
            labels.append(target.cpu().numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    # 确保特征是2D的 (n_samples, n_features)
    if len(features.shape) > 2:
        features = features.reshape(features.shape[0], -1)

    # 应用t-SNE降维
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=2025)
    features_tsne = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    # 为每个类别创建散点
    for label in unique_labels:
        mask = labels == label
        plt.scatter(features_tsne[mask, 0], features_tsne[mask, 1],
                    label=classes_dict[label], alpha=0.6,
                    edgecolors='w', linewidths=0.5)

    # 添加图例和标题
    plt.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc='upper right')
    plt.title(f't-SNE Visualization of {model_name}, SUB: {sub}', fontsize=14)
    plt.grid(True, alpha=0.3)
    # 删除坐标轴
    plt.axis('off')
    # 调整布局
    plt.tight_layout()

    # 保存或显示
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.close()


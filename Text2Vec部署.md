# Text2Vec-Chinese 部署

## 1. Python 依赖

需要安装 transformer 库，text2vec 库非必要

```shell
pip install transformer text2vec -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 2. 导入模型

text2vec-chinese 有三个模型

shibing624/text2vec-base-chinese（原版作者）

GanymedeNil/text2vec-base-chinese

GanymedeNil/text2vec-large-chinese

这里我们以原版模型为例

访问 hugging face 项目主页

https://huggingface.co/shibing624/text2vec-base-chinese

克隆 text2vec-base-chinese 项目

```shell
# 不含模型参数
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/shibing624/text2vec-base-chinese
```

下载模型参数，覆盖到项目目录中

https://huggingface.co/shibing624/text2vec-base-chinese/blob/main/pytorch_model.bin

## 3. 运行测试

使用 Text2Vec 库测试模型能否正常运行

```python
from text2vec import SentenceModel

model = SentenceModel('text2vec-base-chinese')
sentences = ['如何更换花呗绑定银行卡', '花呗更改绑定银行卡', 
             '如何办理银行卡', '哈里路大旋风']

embeddings = model.encode(sentences)
print(embeddings)
```

可视化四个句子的差异（ChatGPT 写的）

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# PCA
pca = PCA(n_components=2)
embeddings_pca = pca.fit_transform(embeddings)

# t-SNE with perplexity=2
tsne = TSNE(n_components=2, random_state=0, perplexity=2)
embeddings_tsne = tsne.fit_transform(embeddings)

# Labels for the sentences
labels = ['1', '2', '3', '4']

# Plot PCA
plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], c='r', marker='o')

for i, txt in enumerate(labels):
    plt.annotate(txt, (embeddings_pca[i, 0], embeddings_pca[i, 1]))

plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('PCA')
plt.axis('equal')

# Plot t-SNE
plt.subplot(122)
plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c='b', marker='o')

for i, txt in enumerate(labels):
    plt.annotate(txt, (embeddings_tsne[i, 0], embeddings_tsne[i, 1]))

plt.xlabel('t-SNE1')
plt.ylabel('t-SNE2')
plt.title('t-SNE')
plt.axis('equal')

plt.tight_layout()
plt.show()
```


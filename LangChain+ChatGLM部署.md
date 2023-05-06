整体流程：

<u>文档通过 pdf 读取 + 分词 + 向量化 保存到数据库中</u>

<u>根据提问检索数据库，找出相关上下文段落</u>

<u>LangChain 将 上下文 + 提问 整合到一起，输入给 ChatGLM</u>

<u>返回字符串，进行输出</u>

参考项目：

LangChain - 打造自己的GPT（二）simple-chatpdf

https://github.com/HappyGO2023/simple-chatpdf

https://zhuanlan.zhihu.com/p/620422560

基于本地知识的 ChatGLM 应用实现

https://github.com/imClumsyPanda/langchain-ChatGLM

## 1. 准备环境

### 1.1 新建环境

创建环境，推荐 python 3.8 或 3.9

```shell
conda activate -n chat python=3.9
```

### 1.2 安装依赖

这里使用了 pip 进行安装

首先先装 torch，直接下载本地文件（需要按照自己的 cuda 版本进行选择）

https://download.pytorch.org/whl/cu117/torch-2.0.0%2Bcu117-cp39-cp39-win_amd64.whl

```python
# 在线安装
pip install torch --index-url https://download.pytorch.org/whl/cu117
# 本地安装（推荐）
pip install torch-2.0.0+cu117-cp39-cp39-win_amd64.whl
```

然后安装 langchain 和 chatglm 需要的库

```shell
# 一键安装
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# 安装的依赖
langchain
PyPDF2
chromadb
protobuf
transformers==4.27.1
cpm_kernels
torch>=1.10
gradio
mdtex2html
sentencepiece
accelerate
```

### 1.3 下载模型

下载 text2vec-base-chinese

https://huggingface.co/shibing624/text2vec-base-chinese

首先 clone 项目目录

```shell
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/shibing624/text2vec-base-chinese
```

然后下载模型文件，放到目录中

https://huggingface.co/shibing624/text2vec-base-chinese/resolve/main/pytorch_model.bin



下载 chatglm-6b-int4

https://huggingface.co/THUDM/chatglm-6b-int4

首先 clone 项目目录

```shell
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/THUDM/chatglm-6b-int4
```

然后下载模型文件，放到目录中

https://cloud.tsinghua.edu.cn/d/674208019e314311ab5c/?p=%2Fchatglm-6b-int4&mode=list

## 2. 模型运行

简化版参考 ChatGLM_QA.ipynb

详细版参考 DocumentQA.ipynb，向量化选择 4.3，模型选择 6.2
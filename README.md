# LangChain-Intro
LangChain 的使用教程，基于 LangChain、OpenAI 官方文档与 HappyGO2023/simple-chatpdf 项目

**文档部分：**

LangChain介绍.md：整体文档，包括使用 OpenAI 和 ChatGLM 对 LangChain 进行使用，以及 LangChain 相关知识和概念的说明

ChatGLM部署.md：简单部署并跑通ChatGLM

LangChain+ChatGLM部署.md：使用 ChatGLM 进行文档问答，搭配程序进行使用

**程序部分：**

DocumentQA.ipynb：使用 LangChain 读取文档进行问答，包括 OpenAI 和 ChatGLM 两种方法

ChatGLM_QA.ipynb：DocumentQA的简化版，使用本地 ChatGLM 模型结合 LangChain 进行问答

Embedding_text2vec.ipynb：使用 text2vec 对本地 Embedding 模型的测试和可视化

**Todo：**

目前正在更新文档问答的自定义prompt部分（LangChain官方使用的是英文prompt）

后续对 pdf 读取进行优化，加入扫描pdf读取（使用OCR）和优化表格读取

考虑将几个太长的 Notebook 进行拆分，分开多个小部分进行介绍

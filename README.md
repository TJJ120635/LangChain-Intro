**Update:2024/08/10**

**本教程是23年编写的，基于 langchain 0.0x 版本。目前 langchain 已经更新到 0.2 版本，代码结构变化较大，可能部分代码不适用。如果要使用本教程，推荐安装 langchain 0.1 版本入手。**

# LangChain-Intro
LangChain 的使用教程，基于 LangChain、OpenAI 官方文档与 HappyGO2023/simple-chatpdf 项目

**文档部分：**（推荐阅读顺序）

1. Text2Vec部署：在本地部署 Text2Vec-Chinese 模型并测试文本向量化
2. ChatGLM部署：在本地部署 ChatGLM-6b 模型并测试问答
3. LangChain入门+部署：使用 LangChain 对 PDF 文档进行提取、检索、问答，搭配 OpenAI 模型或本地模型两种方式
4. LangChain进阶（持续更新）：LangChain 相关知识和概念的说明，文档加载，ChatGLM 使用，自定义 Prompt 替换等

**程序部分：**

1. text2vec_test.ipynb：部署 Text2Vec-Chinese 后进行简单测试和可视化
2. chatglm_test.py：部署 ChatGLM 后进行简单测试
3. LangChain入门.ipynb：搭配 LangChain 入门笔记的简单测试
4. 文档相关.ipynb：搭配 LangChain 进阶笔记的文档相关操作
5. ChatGLM 相关.ipynb：搭配 LangChain 进阶笔记的 ChatGLM 相关操作
6. PDFLoader.py （持续更新）：基于 pypdf 和 TextSplitter 将 PDF 读取和划分封装成头文件
7. ChatGLM.py（持续更新）： 基于 LLM 将 ChatGLM 封装成类
8. QAprompt.py（持续更新）：基于 question_answering.stuff_prompt 自定义中文文档问答 prompt

**Todo：**

后续对 pdf 读取进行优化，加入扫描pdf读取（使用OCR）和优化表格读取

[TOC]



# LangChain 入门+部署

整体流程：

<u>文档通过 pdf 读取 + 分词 + 向量化 保存到数据库中</u>

<u>根据提问检索数据库，找出相关上下文段落</u>

<u>LangChain 将 上下文 + 提问 整合到一起，输入给 ada/davinci/ChatGPT 等语言模型</u>

<u>返回字符串，进行输出</u>

参考项目：

LangChain - 打造自己的GPT（二）simple-chatpdf

https://github.com/HappyGO2023/simple-chatpdf

https://zhuanlan.zhihu.com/p/620422560

基于本地知识的 ChatGLM 应用实现

https://github.com/imClumsyPanda/langchain-ChatGLM

整体框架：（模型方面有 OpenAI 模型或者本地模型两种选择）

流程引擎 - LangChain

文本向量化模型 - OpenAI/text2vec-chinese

语言模型 - OpenAI/ChatGLM



**推荐先使用 OpenAI API 了解 LangChain + LLM 的使用，再上手本地模型**



## 一、准备 Python 环境

### 1.1 新建环境

创建 python 环境

```shell
conda activate -n chat python=3.9
```

### 1.2 安装依赖

conda 太慢，考虑使用 pip 安装

首先先装 torch，直接下载本地文件安装较快（需要按照自己的 CUDA 版本进行选择）

https://download.pytorch.org/whl/cu117/torch-2.0.0%2Bcu117-cp39-cp39-win_amd64.whl

```shell
# 在线安装
pip install torch --index-url https://download.pytorch.org/whl/cu117
# 本地安装（推荐）
pip install torch-2.0.0+cu117-cp39-cp39-win_amd64.whl
```

安装 LangChain 相关依赖，参考 requirements.txt

如果不使用本地模型（ChatGLM + text2vec-chinese）则不需要安装相关的库

```shell
# 一键安装
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# LangChain
torch>=1.10
langchain
pypdf
chromadb
# OpenAI
openai
tiktoken
# 本地模型
transformers==4.27.1
protobuf
cpm_kernels
sentencepiece
accelerate
```

如果后续调用 OpenAI 的时候报错，则需要降级 urllib3

```shell
pip install urllib3==1.25.11
```

## 二、开通 OpenAI API

### 2.1 账号免费额度

LangChain 整合了多种语言模型，包括 OpenAI API、本地模型等

<u>作为实验，我们先使用 OpenAI 的接口，然后再换成本地的模型</u>

<u>每个 OpenAI 账号里会有免费的额度，需要注意检查有没有过期</u>

账号里有 \$5 免费额度，调用 gpt-3.5-turbo \$0.002 / 1K tokens，可以使用很久

下面的提示是没钱的报错：

```
Retrying langchain.llms.openai.completion_with_retry.<locals>._completion_with_retry in 4.0 seconds as it raised RateLimitError: You exceeded your current quota, please check your plan and billing details..
```

### 2.2 不同模型的价格

https://platform.openai.com/account/usage 查询账号的用量

https://platform.openai.com/docs/models/gpt-3-5 查看不同的模型

https://openai.com/pricing#language-models 查看不同的价格

主要有三种模型：GPT-4，GPT-3.5 (Chat)，GPT-3 (InstructGPT)

<u>语言模型可以选 gpt-3.5-turbo (ChatGPT) 或者 ada</u>

<u>Embedding 模型目前只有 ada 开放，不需要选择</u>

| Model         | Price / 1K tokens |
| ------------- | ----------------- |
| gpt-4         | $0.03             |
| gpt-4-32k     | $0.06             |
| gpt-3.5-turbo | $0.002            |
| davinci       | $0.02             |
| curie         | $0.002            |
| babbage       | $0.0005           |
| ada           | $0.0004           |

https://codechina.org/2023/02/openai-gpt-api-summarize/ GPT-3 各模型区别

LangChain 里面的默认模型：

**OpenAI**类默认对应 “text-davinci-003” 版本：

```python3
OpenAI(temperature=0)
```

**OpenAIChat**类默认是 "gpt-3.5-turbo"版本：

```python3
OpenAIChat(temperature=0)
```

## 三、试验普通 LLM（OpenAI 接口）

推荐在 Anaconda 安装 JupyterLab，使用 iPython notebook 运行

### 3.1 设置环境信息

然后填入环境信息，访问 OpenAI 需要挂代理

https://github.com/zhayujie/chatgpt-on-wechat/issues/351

我用 SSR 开启 “允许来自局域网的连接”，将端口设成 1080

网络流向：Python - 1080 端口 - SSR - OpenAI

如果是不同的代理软件，可能需要配置不同端口

```python
import os
# 填入自己的 OpenAI Key
os.environ["OPENAI_API_KEY"] = "Key"
# 设置代理端口
os.environ["HTTP_PROXY"] = "127.0.0.1:1080"
os.environ["HTTPS_PROXY"] = "127.0.0.1:1080"
```

### 3.2 使用普通 LLM 问答

创建一个 LLM（可以使用 ada, davinci 等）

```python
from langchain.llms import OpenAI
llm = OpenAI(model_name="text-ada-001", temperature=0.9)
```

只需要输入你的指令或者问题就可以开始问答

```python
query = "Please introduce yourself."
result = llm(query)
print(result)
```

奇奇怪怪的回答

```
I am a 28-year-old singleartist who is looking for a relationship that is serious and interested in 2-3 years of experience in the artist base.
```

### 3.3 使用 PromptTemplate 和 Chain 问答

设置一个 PromptTemplate，将提问设置成模板，使用 format 从模板创建实例

```python
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["chara"],
    template="Introduce yourself as {chara}",
)

prompt.format(chara="a musician")

'Introduce yourself as a musician in one sentence.'
```

创建一个 chain，将 LLM 和 prompt 结合起来，实现一键调用

```python
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

result = chain.run("the Dragonborn of The Elder Scrolls")
print(result)
```

ada 返回的结果，有点乱

```python
I am the Dragonborn, the invoking and battle-ready! I am the —

Frosty, the Tauren player character who began the trend ofistic skin blue eyes and The Elder ScrollsSSSS

I am the Dragonborn, the invoking and battle-ready! I am the

Tauren, the infant god- alerted by the call of the rising sun! I am

the Tauren, the infant god- alerted by the call of the rising sun!
```

## 四、试验 ChatModels（OpenAI 接口）

普通 LLM 接受一段字符串作为输入，并输出一段字符串

ChatModels 则接受一段聊天信息作为输入，并输出一段字符串

聊天信息中包含了 SystemMessage/HumanMessage/AIMessage

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
```

聊天信息也有相应的 Template 和 Chain，格式相对更复杂

### 4.1 HumanMessage

通过 HumanMessage 和模型进行交互

```python
messages = [HumanMessage(content="请介绍一下你自己.")]
result = chat(messages)
```

```python
content='我是一个人工智能语言模型，被称为OpenAI的GPT-3。我可以回答各种问题，生成文章、对话和其他文本形式的内容。我可以学习和理解不同的语言和主题，并尽可能地回答问题和提供有用的信息。' additional_kwargs={}
```

### 4.2 SystemMessage

通过 SystemMessage 对模型提供对话的背景信息，再用 HumanMessage 做具体问答

```python
messages = [
    SystemMessage(content="You are the Dragonborn in The Elder Scrolls."),
    HumanMessage(content="请介绍一下你自己.")
]
result = chat(messages)
```

```python
content='我是龙裔，又称为龙降临者，是《上古卷轴》系列游戏中的主角。我是一个强大的战士、法师和盗贼，可以使用龙语法术和武器来打败敌人。我的任务是抵御龙的入侵，并拯救天际省的人民。我还可以加入各种组织，如黑暗兄弟会、盗贼公会和战士公会，并完成许多任务和副本。总之，我是一个英勇的冒险家，致力于保护天际省和居民的安全。' additional_kwargs={}
```

注：SystemMessage 是系统告诉 GPT 的信息，可以要求 GPT 作为一个怎么样的身份去交互（你是一个对人类友好的人工智能助手），并且提供一些要求（你的回答需要友好、理性、完善，拒绝不道德的提问）。HumanMessage 是人类用户对 GPT 的问答内容

## 五、simple-chatpdf

参考项目地址

https://github.com/HappyGO2023/simple-chatpdf

embedding.py 进行知识库存储，然后由 qa.py 进行知识库问答

整体流程

1. 把你的内容拆成一块块的小文件块、对块进行了Embedding后放入向量库索引 （为后面提供语义搜索做准备）。

2. 搜索的时候把Query进行Embedding后通过语义检索找到最相似的K个Docs。

3. 把相关的Docs组装成Prompt的Context，基于相关内容进行QA，让GPT进行In Context Learning，用人话回答问题。

### 5.1 PDF 载入

LangChain 里面由 Document loader 和 Text Splitter 两种类型

Document Loader 负责读取各种各样的数据，包括 txt/pdf/markdown 等文档、excel/sql表/dataframe 等结构化数据、网页/Email/PPT 等非结构化数据

Text Splitter 负责将读取到的文档进行清洗和切分，包括按分隔符切分、按语义切分等方式



我们创建一个 loader 和一个 splitter，然后使用 load_and_split 将文档分成若干个小段，每一个小段叫做一个 Document

```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

loader = PyPDFLoader("KOS：2023中国市场招聘趋势.pdf")
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 400,
    chunk_overlap  = 100,
    length_function = len,
)
pages = loader.load_and_split(text_splitter)
```

加载好的 pages 是一个 Document 列表

包括 page_content 文本内容，metadata 信息字段

使用 PyPDFLoader 会自动创建 metadata，包括 source 和 page 两个字段，可以根据需要对 metadata 进行自定义

```python
Document(page_content='xxx', metadata={'source':'xxx, ...'})
```

### 5.2 文档存储

上一部分我们将 pdf 切分成一个个 Document，现在我们希望用本地文件将它们存起来

同时，我们希望每个 Document 有一个索引，根据 query 能匹配到最接近的索引，从而检索到对应的 Document，实现查询功能

LangChain 提供了向量数据库 Vectorstores 模块，轻量化的数据库可以选择 Chroma 或者 FAISS

将文档向量化得到 Embeddings 索引，保存到 Chroma 数据库，然后持久化到本地目录

```python
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

embedding = OpenAIEmbeddings()
persist_directory = 'db_openai'

vectordb = Chroma.from_documents(documents=pages, embedding=embedding, persist_directory=persist_directory)

vectordb.persist()
```

### 5.3 文档读取

加载 Chroma 数据库，创建一个 retriever，作为数据库的检索器

```python
vectordb = Chroma(persist_directory='db_openai', embedding_function=embedding)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})
```

根据 query，用 retriever 查找出最相关的 k 个 Document（段落）

```python
query = '2022年国内新能源车的渗透率是多少?'
docs = retriever.get_relevant_documents(query)
```

### 5.4 文档问答

创建一个 prompt_template，传入上下文和提问，让模型根据上下文回答

需要填入的参数为 context 和 query

```python
from langchain.prompts import PromptTemplate

prompt_template="""请注意：请谨慎评估提问与提示的上下文材料的相关性，只根据本段输入文字信息的内容进行回答，如果提问与提供的上下文材料无关，请回答"我不知道"，不要回答无关答案或编造答案：
-----------------------
上下文: 
    {context}
-----------------------
提问: {query}
-----------------------
答案:"""

prompt = PromptTemplate(
    input_variables=["context", "query"],
    template=prompt_template
)

query = '2022年国内新能源车的渗透率是多少?'
docs = retriever.get_relevant_documents(query)
```

指定一个语言模型，这里使用了 ChatOpenAI

```python
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0)
```

使用 chain 将 LLM 和 prompt 组合到一起

LangChain 中自带 load_qa_chain，可以实现基于 docs 进行问答

我们将 prompt 替换成我们自己的 prompt

```python
from langchain.chains.question_answering import load_qa_chain

chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
```

运行 chain 时填入参数，context = 查找出来的 docs，query = 你的提问

```python
result = chain({"input_documents": docs, "query": query}, 
               return_only_outputs=True)
# result = chain.run(input_documents = docs, query = query)
print(result)
```

## 六、使用本地模型

需要准备好 Text2Vec-Chinese 和 ChatGLM-6b 模型，参考部署文档

### 6.1 本地 Embedding

将 5.2 中的 OpenAI Embedding 替换成本地 Text2Vec Embedding

注意 OpenAI 和本地模型的持久化目录需要分开，不然会出现写入错误

```python
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

embedding = HuggingFaceEmbeddings(model_name='text2vec-base-chinese')
persist_directory = 'db_huggingface'
```

embedding 模型的后续使用和 5.2 完全相同

### 6.2 本地 LLM

将 5.4 中的 ChatOpenAI 模型替换成本地 ChatGLM

```shell
from ChatGLM import ChatGLM

llm = ChatGLM(model_path='chatglm-6b-int4')
```

具体的 ChatGLM 类实现参见 ChatGLM.py
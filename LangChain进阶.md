[TOC]

# LangChain 进阶

## 一、LangChain 的基本概念

### 1.1 Models

模型可以是 OpenAI 接口，也可以是本地部署的模型

LLMs：普通语言模型，输入一段话输出一段话

Chat Models：对话模型，输入输出的格式包括AI信息、系统信息、用户信息等，支持流式回答（一个一个字打出来）

Text Embedding Models：用于文本向量化，输入文本输出一组浮点数

### 1.2 Prompt/PromptTemplate

提示和提示模板，作为模型的输入

模板支持多个信息字段的输入，例如 context + query

其中区分了普通 LLMs 的模板（字符串）和 Chat Models 的模板（ChatMessage）

在模板中还可以加入 few-shot 的例子，整合到 prompt 的前面

### 1.3 Chain

流程链，简单的链就三步：从模板创建 prompt，将 prompt 输入到模型，得到输出。其中输入模型和获取输出是一起的

**chain_type**

stuff：直接将所有 doc 丢给 llm，可能会超 token

map_reduce：每个 doc 进行总结，再做整体总结

refine：总结1+doc2=总结2，总结2+doc3=总结3，......

map_rerank：对每个 doc 计算匹配度，选择最高分数的 doc 给 llm 做回答

### 1.4 Index

提取、存储和检索文档信息

包括 Document Loaders，Text Splitters，Vectorstores，Retrievers 四部分

Document Loaders + Text Splitters 从结构化或非结构化数据提取信息，切分成多个小的 Document，包含文本内容 page_content 和元信息 metadata

Vectorstores 将 Document 向量化得到 Embedding 特征，保存到向量数据库

Retrievers 根据 query 对向量数据库进行检索，匹配最相关的 k 个 Documents

### 1.5 Memory

向 Chain 和 Agent 添加状态，例如短期记忆或长期记忆

将多轮对话的历史保存成 Memory 类

对于用户，可以通过 Memory 提取对话历史信息

对于模型，可以将对话历史整合成 prompt 作为输入

LangChain 会自动将对话历史整理成类似下面的形式，输入到模型：

```python
query = """
[round0]
Human:q1
AI:a1
[round1]
Human:q2
AI:a2
[round2]
Human:q3
AI:"""
result = llm(query)
```

### 1.6 Agent

https://zhuanlan.zhihu.com/p/619344042 详解

LLM作为驾驶员，根据用户的输入动态调用 Chain/Tool

核心概念：

- Tool：例如 google 搜索、数据库检索、Python 等等，可供调用。一般输入输出都是字符串。每个 tool 都有一段语言描述，相当于说明书
- LLM：语言模型
- Agent：需要使用的代理。`zero-shot-react-description`、`react-docstore` 等多种预设类型

`zero-shot-react-description`：根据每个 tools 的描述，选择出需要的 tool

```python
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

agent.run("What was the high temperature in SF yesterday in Fahrenheit? What is that number raised to the .023 power?")
```

```
> Entering new AgentExecutor chain...
 I need to find the temperature first, then use the calculator to raise it to the .023 power.
Action: Search
Action Input: "High temperature in SF yesterday"
Observation: San Francisco Temperature Yesterday. Maximum temperature yesterday: 57 °F (at 1:56 pm) Minimum temperature yesterday: 49 °F (at 1:56 am) Average temperature ...
Thought: I now have the temperature, so I can use the calculator to raise it to the .023 power.
Action: Calculator
Action Input: 57^.023
Observation: Answer: 1.0974509573251117

Thought: I now know the final answer
Final Answer: The high temperature in SF yesterday in Fahrenheit raised to the .023 power is 1.0974509573251117.

> Finished chain.
```

https://liaokong.gitbook.io/llm-kai-fa-jiao-cheng/

![](https://pic4.zhimg.com/v2-1274c950395603397847feb23e2c3a0b_r.jpg)

## 二、文档相关

LangChain 官方封装好了 PyPDFLoader 和 TextSplitter

如果预设的文档加载方式不满足我们的需求，那么就需要自行定义

https://github.com/HappyGO2023/simple-chatpdf

下面就介绍从0开始处理文档的方式

![](D:\文档\笔记\GPT\img\向量数据库.png)

### 2.1 载入 PDF

使用 pypdf.PdfReader 将 pdf 加载进来

```python
import pypdf

pdf_path = 'KOS：2023中国市场招聘趋势.pdf'
pdf_file = open(pdf_path, 'rb')
pdf_reader = pypdf.PdfReader(pdf_file)

pdf_content = ''
for num in range(len(pdf_reader.pages)):
    page = pdf_reader.pages[num]
    pdf_content += page.extract_text()
```

pdf_reader.pages[x] 就是每一页的内容，包括了很多格式之类的信息

pdf_reader.pages[x].extract_text() 将每一页的文字提取出来，存入 pdf_cotent 字符串

一个 89 页的 PDF 可以有 34400 字

### 2.2 文档清洗

全文档字符串 >> 分句 >> 多句合并成段

pdf_content 将所有内容保存成一个字符串

首先需要将换行洗掉

然后按照句子结束符号（；。！？等），拆分成一句句话

```python
pdf_content = pdf_content.replace('\n', '') 
pdf_content = pdf_content.replace('\n\n', '') 
pdf_content = re.sub(r'\s+', ' ', pdf_content)

pdf_sentences_mark = re.split('(；|。|！|\!|\.|？|\?)', pdf_content) 

pdf_sentences = []
for i in range(int(len(pdf_sentences_mark)/2)):
    sent = pdf_sentences_mark[2*i] + pdf_sentences_mark[2*i+1]
    pdf_sentences.append(sent)
if len(pdf_sentences_mark) % 2 == 1:
    pdf_sentences.append(pdf_sentences_mark[len(pdf_sentences_mark)-1])
```

接下来将多个句子拼成一段，按照最大长度 300 为一段

最后得到一个 paragraphs，list 类型，每个元素为一段字符串，len < 300

这样我们就把完整的 PDF 拆分成了一段段小文本

也可以用 langchain.text_splitter.CharacterTextSplitter 来尝试

```python
paragraphs = []
max_len = 300
current_len = 0
current_para = ""

for sent in pdf_sentences:
    sent_len = len(sent)
    if current_len + sent_len <= max_len:
        current_para += sent
        current_len += sent_len
    else:
        paragraphs.append(current_para.strip())
        current_para = sent
        current_len = sent_len

paragraphs.append(current_para.strip())
```

### 2.3 文档保存成 Documents

文字需要保存成 langchain.docstore.document.Document 类型

分好的每一段是一个小 Document

包含 page_content 文本内容，metadata 是自己填写的信息字段，用于数据库检索

```python
Document(page_content='xxx', metadata={'source':'xxx, ...})
```

把每个小 Document 放进一个 documents list 里

整个 documents[ ] 就是完整的 pdf

```python
from langchain.docstore.document import Document

documents = []
metadata = {"source": pdf_path}
for para in paragraphs:
    new_doc = Document(page_content=para, metadata=metadata)
    documents.append(new_doc)
```

如此我们就从一个文档得到了一个 Documents 列表，可以导入到向量数据库中

### 2.4 多文档导入向量数据库

在入门篇中我们通过 from_documents 命令可以从一个文档生成一个向量数据库

如果我们想在现有的数据库中添加新的文档，则需要用 add_documents 方法

这里换成了 FAISS 数据库，使用方式和 Chroma 相同，LangChain 整合好了接口

首先需要新建一个数据库，或者使用现有的数据库

```python
from langchain.vectorstores import FAISS

# 新建数据库
loader = PyPDFLoader(file_path + file_list[0])
docs = loader.load_and_split(text_splitter)
vectordb = FAISS.from_documents(docs, db_embedding)
# 使用现有数据库
vectordb = FAISS.load_local(embeddings=db_embedding, folder_path=db_directory)
```

接下来向数据库中添加新的文档

```python
loader = PyPDFLoader(file_path + file)
docs = loader.load_and_split(text_splitter)
vectordb.add_documents(docs)
```

最后再重新持久化

```python
vectordb.save_local(db_directory)
```

### 2.5 PDF 封装

在 2.1 - 2.3 介绍了怎么将文档从0变成一个 Documents 列表，接下来需要将 PDF 读取的过程封装成函数，分为两种方式

第一种是使用官方默认的 langchain.document_loaders.PyPDFLoader + 自定义的 PDFTextSplitter，通过 load_and_split 实现划分

第二种是直接实现自定义的 PDFLoader 函数，实现载入、清洗、保存

#### 2.5.1 PDFSplitter

LangChain 会首先使用 Loader 读取文档，再使用 Splitter 进行划分

自定义的 PDFTextSplitter 类需要继承 TextSplitter 基类，实现的核心是 split_text 函数

这里我们定义了三个规则 1. 多个连续空格保留一个空格 2. 多个连续换行 + 空格的组合保留一个换行 3. 多个重复标点符号保留一个

利用 re 对文本进行清洗后，基于分隔符（默认为 '\n'）使用基类的 split 和 _merge_splits 方法进行句子的分割与段落合并

这里直接将 '\n' 作为分隔符，而没有考虑使用中文标点来进行断句。我的考虑是按行切分能够将标题区分开，模型也能正确识别，如果按标点符号切分则会导致标题和正文混在一起

```python
import re
from langchain.text_splitter import TextSplitter
from typing import (
    Any,
    List,
)

class PDFTextSplitter(TextSplitter):
    def __init__(self, separator: str = "\n", **kwargs: Any):
        super().__init__(**kwargs)
        self._separator = separator

    def split_text(self, text: str) -> List[str]:
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\s*\n\s*', '\n', text)
        text = re.sub(r'([，。？！…；：,.?!;:])\1+', r'\1', text)

        if self._separator:
            splits = text.split(self._separator)
        else:
            splits = list(text)
        return self._merge_splits(splits, self._separator)
```

使用的时候参考基类 TextSplitter 的使用，可以传入 chunk_size, chunk_overlap, length_function, separator 等参数

```python
text_splitter = PDFTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    length_function=len,
)
```

#### 2.5.2 PDFSplitter

第二种方法是直接将读取 + 切分的整个过程封装起来，和 2.1-2.3 类似

```python
import os
import re
from pypdf import PdfReader
from langchain.docstore.document import Document


def PDFLoader(pdf_path: str, max_len: int = 300) -> list:
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PdfReader(pdf_file)

        # Extract PDF content
        pdf_content = ''.join(page.extract_text() for page in pdf_reader.pages)

        # Clean up symbols
        pdf_content = re.sub(r'\n+', ' ', pdf_content)
        pdf_content = re.sub(r'\s+', ' ', pdf_content)

        # Split into sentences
        sentence_separator_pattern = re.compile('([；。！! \?？]+)')
        sentences = [
            element
            for element in sentence_separator_pattern.split(pdf_content)
            if not sentence_separator_pattern.match(element) and element
        ]

        # Merge sentences into paragraphs
        paragraphs = []
        current_length = 0
        current_paragraph = ""

        for sentence in sentences:
            sentence_length = len(sentence)
            if current_length + sentence_length <= max_len:
                current_paragraph += sentence
                current_length += sentence_length
            else:
                paragraphs.append(current_paragraph.strip())
                current_paragraph = sentence
                current_length = sentence_length

        paragraphs.append(current_paragraph.strip())
        
        documents = []
        pdf_dir, pdf_name = os.path.split(pdf_path)
        metadata = {"directory":pdf_dir , "source": pdf_name}
        for para in paragraphs:
            new_doc = Document(page_content=para, metadata=metadata)
            documents.append(new_doc)

    return documents
```

## 三、ChatGLM 相关

### 3.1 ChatGLM 类封装

尽管 LangChain 支持了 OpenAI、LLaMA、GPT4ALL、Hugging Face 等多种模型，但是没有预设的 ChatGLM 类。因此需要自己创建一个类

类的实现参考项目中的 models/chatllm.py

https://github.com/imClumsyPanda/langchain-ChatGLM

基于 LangChain 的 LLM 基类，创建 ChatGLM 类

构造函数 \_\_init\_\_ 参考 ChatGLM 官方的模型加载，使用 transformers 库和 AutoModel

析构函数 \_\_del\_\_ 使用 torch.cuda.empty_cache() 手动回收显存，否则只有关闭程序时显存才会释放

调用函数 _call 根据 prompt 输出回答，输入输出都是字符串

（具体的方法参考 8.8 Custom LLM）

```python
# ChatGLM.py
from typing import Any, List, Mapping, Optional
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModel, AutoConfig

import torch


def torch_gc():
    # with torch.cuda.device(DEVICE):
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    

class ChatGLM(LLM):
    max_token: int = 10000
    temperature: float = 0.01
    top_p = 0.9
    tokenizer: object = None
    model: object = None

    def __init__(
        self,
        model_path: str = "chatglm-6b-int4",
        **kwargs
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True, 
            revision=model_path
        )
        model_config = AutoConfig.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            revision=model_path
        )
        self.model = AutoModel.from_pretrained(
            model_path, 
            config=model_config, 
            trust_remote_code=True, 
            revision=model_path, 
            **kwargs
        ).half().cuda()
        self.model = self.model.eval()

    def __del__(self):
        self.tokenizer = None
        self.model = None

        torch.cuda.empty_cache()        
    
    @property
    def _llm_type(self) -> str:
        return "ChatGLM"

    def _call(
        self, 
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        
        response, _ = self.model.chat(
            self.tokenizer,
            prompt,
            history=[],
            max_length=self.max_token,
            temperature=self.temperature,
        )
        torch_gc()
        return response
```

### 3.2 ChatGLM 使用

在封装好 ChatGLM 类之后，使用的方式和 OpenAI 模型类似

声明时需要传入模型的路径

```python
llm = OpenAI()
llm = ChatGLM(model_path='../chatglm-6b-int4')
```

由于 ChatGLM 封装使用的是 LLM 基类，而不是 ChatModel，调用只需要传入字符串

```python
result = llm('你好')
print(result)

你好，我是 ChatGLM-6B，是清华大学KEG实验室和智谱AI公司于2023年共同训练的语言模型。我的任务是服务并帮助人类，但我并不是一个真实的人。
```

### 3.3 模型问答+对话历史

对话历史使用 ConversationBufferMemory

根据向量知识库回答使用 ConversationalRetrievalChain

#### 3.3.1 ConversationBufferMemory

模型加入对话历史的实现方式其实很简单，只需要在 prompt 前面加上每一轮的人类输出和AI输出即可。LangChain 官方则提供了 Memory 功能，能够自动整合历史记录，不需要自己将对话记录拼接成 prompt

使用的时候首先需要创建 memory，然后向 chain 里引入 memory

```python
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationChain

llm = OpenAI(temperature=0)
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm, 
    verbose=True, 
    memory=memory
)

conversation.predict(input="Hi there!")
```

#### 3.3.2 ConversationalRetrievalChain

https://python.langchain.com/en/latest/modules/chains/index_examples/chat_vector_db.html

这个 chain 和 RetrievalQAChain 差不多，都有基于向量数据库进行问答的功能，但是加上了对话历史

使用也非常简单（其中 llm 和 retriever 都是之前定义好的）

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chain = ConversationalRetrievalChain.from_llm(llm, retriever, memory=memory)
```

在调用时，将提问作为 question 参数传入 chain

返回的结果会包括 question/chat_history/answer 三部分内容

其中对话历史部分是 ChatHistory 类型，由 SystemMessage/HumanMessage/AIMessage 组成

```python
query = '请简单地介绍一下虚拟歌手洛天依'
result = chain({"question": query})

{'question': '请简单地介绍一下虚拟歌手洛天依',
 'chat_history': [HumanMessage(content='请简单地介绍一下虚拟歌手洛天依', additional_kwargs={}, example=False),
  AIMessage(content='洛天依是一个由VOCALOID语音合成引擎软件为基础创造出来的虚拟人物，由真人提供声源，再由软件合成人声，能够让粉丝深度参与共创。从2012年7月12日洛天依出道至今，音乐人以及粉丝已为洛天依创作了超过一万首作品，通过为用户提供更多想象和创作空间的同时，与粉丝建立了更深刻联系。洛天依的出道引起了全球范围内的关注，并成为了许多粉丝的虚拟偶像。', additional_kwargs={}, example=False)],
 'answer': '洛天依是一个由VOCALOID语音合成引擎软件为基础创造出来的虚拟人物，由真人提供声源，再由软件合成人声，能够让粉丝深度参与共创。从2012年7月12日洛天依出道至今，音乐人以及粉丝已为洛天依创作了超过一万首作品，通过为用户提供更多想象和创作空间的同时，与粉丝建立了更深刻联系。洛天依的出道引起了全球范围内的关注，并成为了许多粉丝的虚拟偶像。'}
```

#### 3.3.3 ConversationalRetrievalChain + Source

我们已经使用 chain 实现了基于数据库的问答，并加入了对话历史功能。接下来还可以在返回结果中加上依据的上下文

因此需要在声明 memory 的时候加上 return_messages=True，并指定输出字段 answer 和记忆字段 chat_history。这样就能在回答中加上 source_documents 字段

```python
memory = ConversationBufferMemory(output_key='answer', memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(llm, retriever, memory=memory, return_source_documents=True)
```

```python
query = '请简单地介绍一下虚拟歌手洛天依'
result = qa({"question": query})

{'question': '请简单地介绍一下虚拟歌手洛天依',
 'chat_history': [HumanMessage(content='请简单地介绍一下虚拟歌手洛天依', additional_kwargs={}, example=False),
  AIMessage(content='洛天依是一个由VOCALOID语音合成引擎软件为基础创造出来的虚拟人物，由真人提供声源，再由软件合成人声，能够让粉丝深度参与共创。从2012年7月12日洛天依出道至今，音乐人以及粉丝已为洛天依创作了超过一万首作品，通过为用户提供更多想象和创作空间的同时，与粉丝建立了更深刻联系。洛天依的出道引起了全球范围内的关注，并成为了许多粉丝的虚拟偶像。', additional_kwargs={}, example=False)],
 'answer': '洛天依是一个由VOCALOID语音合成引擎软件为基础创造出来的虚拟人物，由真人提供声源，再由软件合成人声，能够让粉丝深度参与共创。从2012年7月12日洛天依出道至今，音乐人以及粉丝已为洛天依创作了超过一万首作品，通过为用户提供更多想象和创作空间的同时，与粉丝建立了更深刻联系。洛天依的出道引起了全球范围内的关注，并成为了许多粉丝的虚拟偶像。',
 'source_documents': [Document(page_content='VOCALOID语音合成引擎软件为基础创造出来的虚拟人物，由真人提供声源，再由软件合成人声，都是能够让粉丝深度参与共创的虚拟歌手以洛天依为例，任何人通过声库创作词曲，都能达到“洛天依演唱一首歌”的效果从2012年7月12日洛天依出道至今十年的时间内，音乐人以及粉丝已为洛天依创作了超过一万首作品，通过为用户提供更多想象和创作空间的同时，与粉丝建立了更深刻联系二是通过', metadata={'source': '../files/人工智能生成内容白皮书-2022.pdf'}),
  Document(page_content='如欧莱雅、飞利浦、完美日记等品牌的虚拟主播一般会在凌晨0点上线，并进行近9个小时的直播，与真人主播形成了24小时无缝对接的直播服务二是虚拟化的品牌主播更能加速店铺或品牌年轻化进程，拉近与新消费人群的距离，塑造元宇宙时代的店铺形象，未来可通过延展应用到元宇宙中更多元的虚拟场景，实现多圈层传播如彩妆品牌“卡姿兰”推出自己的品牌虚拟形象，并将其引入直播间作为其天猫旗舰店日常的虚拟主播导购', metadata={'source': '../files/人工智能生成内容白皮书-2022.pdf'})]}
```

## 四、中文 prompt 替换

### 4.1 默认 prompt 原理

在 3.3 的问答测试中我们使用的是 ConversationalRetrievalChain，它会使用问答 prompt 将多篇 context 和 query 整合到一起，让模型阅读上下文材料并根据材料作出回答

由于默认 prompt 是英文的，但是材料和问题都是中文的，导致 ChatGLM 会出现回答时中英混乱的情况（ChatGPT 则没有这个问题）

因此我们需要用自定义的 prompt 改写原来的 prompt

ConversationalRetrievalChain 由两个 chain 结合成

其中需要我们注意的参数是 load_qa_chain 的 combine_docs_chain_kwargs

```python
# langchain\chains\conversational_retrieval\base.py
class ConversationalRetrievalChain(BaseConversationalRetrievalChain):
@classmethod
    def from_llm(
        condense_question_prompt: BasePromptTemplate = CONDENSE_QUESTION_PROMPT,
        chain_type: str = "stuff",
        combine_docs_chain_kwargs: Optional[Dict] = None,
        # ......其他参数 
    ) -> BaseConversationalRetrievalChain:
        doc_chain = load_qa_chain(
            llm,
            chain_type=chain_type,
            verbose=verbose,
            **combine_docs_chain_kwargs,
        )
        condense_question_chain = LLMChain(
            llm=llm, prompt=condense_question_prompt, verbose=verbose
        )
        return cls(
            retriever=retriever,
            combine_docs_chain=doc_chain,
            question_generator=condense_question_chain,
            **kwargs,
        )
```

通过 load_qa_chain 将多个文本整合成 context，并加上文档问答的 prompt

这里看到默认的 prompt 是 stuff_prompt.PROMPT_SELECTOR.get_prompt(llm)

```python
# langchain\chains\question_answering\__init__.py
from langchain.chains.question_answering import stuff_prompt

def _load_stuff_chain(
    llm: BaseLanguageModel,
    prompt: Optional[BasePromptTemplate] = None,
    document_variable_name: str = "context",
    verbose: Optional[bool] = None,
    callback_manager: Optional[BaseCallbackManager] = None,
    **kwargs: Any,
) -> StuffDocumentsChain:
    _prompt = prompt or stuff_prompt.PROMPT_SELECTOR.get_prompt(llm)
    llm_chain = LLMChain(
        llm=llm, prompt=_prompt, verbose=verbose, callback_manager=callback_manager
    )
    # TODO: document prompt
    return StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name=document_variable_name,
        verbose=verbose,
        callback_manager=callback_manager,
        **kwargs,
    )
```

stuff_prompt 使用一个 PROMPT.SELECTOR 来进行选择

如果是普通 LLM，那么就使用 PromptTemplate，使用文本 prompt

如果是 ChatModel，那么就使用 ChatPromptTemplate，使用 SystemMessage + HumanMessage

我们可以看到 prompt 的关键就算这段话，让模型根据 context 来进行回答

```python
# langchain\chains\question_answering\stuff_prompt.py

prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

system_template = """Use the following pieces of context to answer the users question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
----------------
{context}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)


PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=PROMPT, conditionals=[(is_chat_model, CHAT_PROMPT)]
)
```

### 4.2 自定义 prompt 编写

到这里我们就了解了 QA prompt 的运作过程，我们可以自己写一个中文版来代替

```python
# QAprompt.py
# flake8: noqa
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

prompt_template = """阅读给出的文本材料，提取有用的信息，对最后的问题进行回答。如果不知道答案，请回答'我不知道'，请不要编造虚假的答案，并保持回答的准确性。
文本材料：
{context}

问题: {question}
有用的回答:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

system_template = """阅读给出的文本材料，提取有用的信息，对最后的问题进行回答。 
如果你不知道答案，请回答'我不知道'，请不要编造虚假的答案，并保持回答的准确性。
----------------
文本材料：
{context}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)


PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=PROMPT, conditionals=[(is_chat_model, CHAT_PROMPT)]
)
```

在使用的时候我们向 ConversationalRetrievalChain 填入新的 QAprompt 即可

```python
memory = ConversationBufferMemory(
    output_key='answer', 
    memory_key="chat_history", 
    return_messages=True)
chain = ConversationalRetrievalChain.from_llm(
    llm, retriever, memory=memory, 
    return_source_documents=True, 
    combine_docs_chain_kwargs={'prompt':QAprompt.PROMPT_SELECTOR.get_prompt(llm)})
```

## 五、Document QA 详细说明

### 5.1 Indexes 总介绍

LangChain 基于 index 对 documents 做匹配

最基本的接口就是 retriever，根据 query 找到最相关的 k 个 documens

对于非结构化数据，例如文档，可以用 index 和 retrieve 来索引

而对于结构化数据，例如 csv，有相应数据类型的索引函数

主要的部分：

Document Loaders

Text Splitters

Vector Stores

Retrievers

### 5.2 Document Loaders

https://python.langchain.com/en/latest/modules/indexes/document_loaders.html

读取支持各种各样的文件格式

格式化文件：json，csv，dataframe，directory，markdown

多媒体：HTML，images，pdf，PPT

其他：bilibili，discord，email，git，gitbook，youtube



对于结构数据，像 csv、dataframe、数据库，一行就是一个 Document，page_content 保存行的主键，metadata 保存每一行的各个值

对于文档文件，像 HTML、markdown、PDF，需要分块成多个段落，每个段落就是一个 Document

对于非结构化数据，像 图片、Youtube 链接，还没弄明白怎么加载



官方预设了多种 Loader，每个 Loader 整合了第三方的库，例如 BS4、PyPDF 等，需要另外安装对应的库。可以自己使用这些库读取好文本，也可以直接用 Loader 加载

### 5.3 Text Splitter

如果用 Loader 加载结构化数据，一行一个 Doc，我们不需要自己进行分段

但是如果读取非结构化数据，例如一个 100 页的 PDF，我们不可能把全部内容一次性丢给 LLM



因此我们希望每一个 Doc 保存其中几页的内容，在用户进行提问的时候找到最相关的几个 Docs，让模型根据这几段内容进行回答

粗暴的划分方式是，直接按照长度将文章切开，可能会导致信息断开。理想的划分方式是，将相关的内容划分到同一个 Doc 里，保持上下文的完整性，但是需要对文章的语义做理解

（看 LangChain 官方文档似乎也没有根据上下文进行分割，只是做了简单的长度分段）



第一种划分方式，按照长度进行划分。将全篇文档拆成一句一句话，每次选择 N 句话组成一段，每段的总字数不超过 max_len

第二种划分方式，按照文档信息进行划分。如果文档自己有章节信息，则找到里面的章节进行划分

第三种划分方式，使用 Text Splitter 进行划分：

（按照知乎评论，默认 Splitter 对中文的理解能力不好。如果采用 Splitter 方式需要另外找中文适配的模型。不知道按长度划分方式和按 Splitter 划分方式的效果差别有多大）

LangChain 里整合了多种 Text Splitter，对文章进行分段，有 chunk_size/chunk_overlap 等参数

```python
from langchain.text_splitter import CharacterTextSplitter

with open("../../state_of_the_union.txt") as f:
    state_of_the_union = f.read()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(state_of_the_union)

docsearch = Chroma.from_texts(texts, embeddings, metadatas=[{"source": str(i)} for i in range(len(texts))]).as_retriever()
```

 默认的选项就算 RecursiveCharacterTextSplitter

根据换行 '\\n', '\\n\\n' 等，将文章分段，假设 chunk_size = 100，chunk_overlap = 10

如果一个段小于 chunk 大小，则在一个 chunk 中装下尽可能多的段。例如三个段 30/65/20，chunk1 = 段 1+2，chunk2 = 段 3 

如果一个段大于 chunk 大小，那就分成两块，后面 chunk 的开头等于前面 chunk 的结尾

```python
# 原文
'Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans.  \n\nLast year COVID-19 kept us apart. This year we are finally together again. \n\n'

# 第一段太长，拆分成两个 chunk
Document(page_content='Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and', metadata={})
Document(page_content='of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans.', metadata={})
# 第二段完整放进 chunk 里
Document(page_content='Last year COVID-19 kept us apart. This year we are finally together again.', metadata={})
```

对于 chunk 长度计算，可以简单使用字符数，也可以使用 huggingface_tokenizer

对于 Latex，有 LatexTextSplitter，可以根据 \\section \\subsection 之类的标识符来划分

对于 Markdown，有 MarkdownTextSplitter，根据 # ## 等划分

还有像 Python 代码等类型，也有智能划分工具

### 5.4 创建 Index

使用预设 Loader 加载数据，如果不需要手动划分，可以创建对应的索引

让 query 在索引里进行匹配

```python
from langchain.indexes import VectorstoreIndexCreator
index = VectorstoreIndexCreator().from_loaders([loader])

query = "What did the president say about Ketanji Brown Jackson"
index.query(query)
```

### 5.5 创建 Chain

两种选项：

with Sources：在回答里附上来源 Document，似乎是 ChatModel

Retrieval：基于 VectorDB，但是没搞懂有什么不同

```python
from langchain.chains.question_answering import load_qa_chain
chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")

from langchain.chains.qa_with_sources import load_qa_with_sources_chain
chain = load_qa_with_sources_chain(OpenAI(temperature=0), chain_type="stuff")

from langchain.chains import RetrievalQA
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever())

from langchain.chains import RetrievalQAWithSourcesChain
chain = RetrievalQAWithSourcesChain.from_chain_type(OpenAI(temperature=0), chain_type="stuff", retriever=docsearch.as_retriever())
```

选择 chain 并运行

```python
from langchain.chains.question_answering import load_qa_chain
chain = load_qa_chain(llm, chain_type="stuff")
chain.run(input_documents=docs, question=query)

from langchain.chains.qa_with_sources import load_qa_with_sources_chain
chain = load_qa_with_sources_chain(llm, chain_type="stuff")
chain({"input_documents": docs, "question": query}, return_only_outputs=True)
```

## 六、草稿区

架构图

https://zhuanlan.zhihu.com/p/613842066

IBM 模型

https://zhuanlan.zhihu.com/p/627449559

## 七、Prompt（待补充）

对 QA 机器人的 prompt 管理，很重要

https://github.com/dair-ai/Prompt-Engineering-Guide

对于 Tool 需要文字相关的说明

对于 Agent 需要流程指引


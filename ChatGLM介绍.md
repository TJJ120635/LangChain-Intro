## 1. 准备工作

安装 CUDA & CuDNN

安装文件

https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#removing-cuda-toolkit-and-driver

https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local

安装指引

https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#verify

https://developer.nvidia.cn/rdp/cudnn-download

新建 conda 环境

```shell
conda create -n chat python=3.8
```

安装 pytorch

```shell
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

克隆项目

```shell
git clone https://github.com/THUDM/ChatGLM-6B.git
cd ChatGLM-6B
```

安装依赖

```shell
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

## 2. 导入模型

（在 ChatGLM-6B 目录里）

克隆 hugging face 的 chatglm-6b

```shell
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/THUDM/chatglm-6b
```

然后在 https://cloud.tsinghua.edu.cn/d/674208019e314311ab5c/?p=%2Fchatglm-6b&mode=list

下载 FP16 的模型参数，丢进 chatglm-6b 文件夹



克隆 hugging face 的 chatglm-6b-int4

```shell
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/THUDM/chatglm-6b-int4
```

然后在 https://cloud.tsinghua.edu.cn/d/674208019e314311ab5c/?p=%2Fchatglm-6b-int4&mode=list

下载 INT4 的模型参数，丢进 chatglm-6b-int4 文件夹



## 3. 运行 test.py(CPU)

按照 GitHub 官方文档“代码调用”章节进行测试

只是使用了 CPU 推理，速度非常慢

```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("chatglm-6b-int4", trust_remote_code=True)
model = AutoModel.from_pretrained("chatglm-6b-int4",trust_remote_code=True).float()
model = model.eval()

print("Begin...")
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
response, history = model.chat(tokenizer, "请介绍一下你自己", history=history)
print(response)
```

![](D:\文档\笔记\GPT\GLM\2.png)



## 4. 修复 CUDA 库（可以跳过）

### 4.1 问题出现

确认 cuda 可用

![](D:\文档\笔记\GPT\GLM\3.png)

修改 test.py，修改模型部分并运行

```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("chatglm-6b-int4", trust_remote_code=True)
model = AutoModel.from_pretrained("chatglm-6b-int4",trust_remote_code=True).half().cuda()
model = model.eval()

print("First:")
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
print("Second:")
response, history = model.chat(tokenizer, "请介绍一下你自己", history=history)
print(response)
```

模型运行时出现报错：

```shell
File "/root/anaconda3/envs/chat/lib/python3.8/site-packages/cpm_kernels/kernels/base.py", line 24, in get_module
    self._module[curr_device] = cuda.cuModuleLoadData(self._code)
  File "/root/anaconda3/envs/chat/lib/python3.8/site-packages/cpm_kernels/library/base.py", line 72, in wrapper
    raise RuntimeError("Library %s is not initialized" % self.__name)
RuntimeError: Library cuda is not initialized
```

![](D:\文档\笔记\GPT\GLM\4.png)

完整报错：

```shell
Explicitly passing a `revision` is encouraged when loading a model with custom code to ensure no malicious code has been contributed in a newer revision.
Explicitly passing a `revision` is encouraged when loading a configuration with custom code to ensure no malicious code has been contributed in a newer revision.
Explicitly passing a `revision` is encouraged when loading a model with custom code to ensure no malicious code has been contributed in a newer revision.
No compiled kernel found.
Compiling kernels : /root/.cache/huggingface/modules/transformers_modules/chatglm-6b-int4/quantization_kernels_parallel.c
Compiling gcc -O3 -fPIC -pthread -fopenmp -std=c99 /root/.cache/huggingface/modules/transformers_modules/chatglm-6b-int4/quantization_kernels_parallel.c -shared -o /root/.cache/huggingface/modules/transformers_modules/chatglm-6b-int4/quantization_kernels_parallel.so
Load kernel : /root/.cache/huggingface/modules/transformers_modules/chatglm-6b-int4/quantization_kernels_parallel.so
Setting CPU quantization kernel threads to 2
Parallel kernel is not recommended when parallel num < 4.
Using quantization cache
Applying quantization to glm layers
First:
The dtype of attention mask (torch.int64) is not bool
Traceback (most recent call last):
  File "/data1/chat_model/test.py", line 14, in <module>
    response, history = model.chat(tokenizer, "你好", history=[])
  File "/root/anaconda3/envs/chat/lib/python3.9/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
  File "/root/.cache/huggingface/modules/transformers_modules/chatglm-6b-int4/modeling_chatglm.py", line 1286, in chat
    outputs = self.generate(**inputs, **gen_kwargs)
  File "/root/anaconda3/envs/chat/lib/python3.9/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
  File "/root/anaconda3/envs/chat/lib/python3.9/site-packages/transformers/generation/utils.py", line 1452, in generate
    return self.sample(
  File "/root/anaconda3/envs/chat/lib/python3.9/site-packages/transformers/generation/utils.py", line 2468, in sample
    outputs = self(
  File "/root/anaconda3/envs/chat/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/root/.cache/huggingface/modules/transformers_modules/chatglm-6b-int4/modeling_chatglm.py", line 1191, in forward
    transformer_outputs = self.transformer(
  File "/root/anaconda3/envs/chat/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/root/.cache/huggingface/modules/transformers_modules/chatglm-6b-int4/modeling_chatglm.py", line 997, in forward
    layer_ret = layer(
  File "/root/anaconda3/envs/chat/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/root/.cache/huggingface/modules/transformers_modules/chatglm-6b-int4/modeling_chatglm.py", line 627, in forward
    attention_outputs = self.attention(
  File "/root/anaconda3/envs/chat/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/root/.cache/huggingface/modules/transformers_modules/chatglm-6b-int4/modeling_chatglm.py", line 445, in forward
    mixed_raw_layer = self.query_key_value(hidden_states)
  File "/root/anaconda3/envs/chat/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/root/.cache/huggingface/modules/transformers_modules/chatglm-6b-int4/quantization.py", line 375, in forward
    output = W8A16Linear.apply(input, self.weight, self.weight_scale, self.weight_bit_width)
  File "/root/anaconda3/envs/chat/lib/python3.9/site-packages/torch/autograd/function.py", line 506, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
  File "/root/.cache/huggingface/modules/transformers_modules/chatglm-6b-int4/quantization.py", line 53, in forward
    weight = extract_weight_to_half(quant_w, scale_w, weight_bit_width)
  File "/root/.cache/huggingface/modules/transformers_modules/chatglm-6b-int4/quantization.py", line 274, in extract_weight_to_half
    func(
  File "/root/anaconda3/envs/chat/lib/python3.9/site-packages/cpm_kernels/kernels/base.py", line 48, in __call__
    func = self._prepare_func()
  File "/root/anaconda3/envs/chat/lib/python3.9/site-packages/cpm_kernels/kernels/base.py", line 40, in _prepare_func
    self._module.get_module(), self._func_name
  File "/root/anaconda3/envs/chat/lib/python3.9/site-packages/cpm_kernels/kernels/base.py", line 24, in get_module
    self._module[curr_device] = cuda.cuModuleLoadData(self._code)
  File "/root/anaconda3/envs/chat/lib/python3.9/site-packages/cpm_kernels/library/base.py", line 72, in wrapper
    raise RuntimeError("Library %s is not initialized" % self.__name)
RuntimeError: Library cuda is not initialized
```

### 4.2 问题定位

/root/anaconda3/envs/chat/lib/python3.9/site-packages/cpm_kernels/library/base.py

```python
def unix_find_lib(name):
    cuda_path = os.environ.get("CUDA_PATH", None)
    if cuda_path is not None:
        lib_name = os.path.join(cuda_path, "lib64", "lib%s.so" % name)
        if os.path.exists(lib_name):
            return lib_name

    cuda_path = "/usr/local/cuda"
    if cuda_path is not None:
        lib_name = os.path.join(cuda_path, "lib64", "lib%s.so" % name)
        if os.path.exists(lib_name):
            return lib_name

class Lib:
    def __init__(self, name):
        self.__name = name
        if sys.platform.startswith("win"):
            lib_path = windows_find_lib(self.__name)
            self.__lib_path = lib_path
            if lib_path is not None:
                self.__lib = ctypes.WinDLL(lib_path)
            else:
                self.__lib = None
        elif sys.platform.startswith("linux"):
            lib_path = unix_find_lib(self.__name)
            # Edit Here
            print(name, ':', lib_path)
            
            self.__lib_path = lib_path
            if lib_path is not None:
                self.__lib = ctypes.cdll.LoadLibrary(lib_path)
            else:
                self.__lib = None
        else:
            raise RuntimeError("Unknown platform: %s" % sys.platform)
```

![](D:\文档\笔记\GPT\GLM\6.png)

在模型加载前首先会搜索 /usr/local/cuda/lib64，导入 .so 库

lib64 中缺少了 libcuda.so，因此报错

### 4.3 问题解决方案

首先搜索现有的 libcuda.so

使用 nvidia-smi 查看驱动版本是 515.65.01

发现在 /usr/lib/x86_64-linux-gnu 中有 libcuda.so.515.65.01

```shell
(base) root@ubuntu-virtual-machine:~# find / -name "libcuda.*"
/usr/local/cuda-11.7/targets/x86_64-linux/lib/stubs/libcuda.so
/usr/lib/x86_64-linux-gnu/libcuda.so
/usr/lib/x86_64-linux-gnu/libcuda.so.1
/usr/lib/x86_64-linux-gnu/libcuda.so.525.105.17
/usr/lib/x86_64-linux-gnu/libcuda.so.515.65.01
/usr/lib/i386-linux-gnu/libcuda.so
/usr/lib/i386-linux-gnu/libcuda.so.1
/usr/lib/i386-linux-gnu/libcuda.so.515.65.01


(base) root@ubuntu-virtual-machine:~# ll /usr/lib/x86_64-linux-gnu/libcuda*
lrwxrwxrwx 1 root root       29 3月  27 19:40 /usr/lib/x86_64-linux-gnu/libcudadebugger.so.1 -> libcudadebugger.so.525.105.17
-rw-r--r-- 1 root root 10490248 3月  27 19:40 /usr/lib/x86_64-linux-gnu/libcudadebugger.so.525.105.17
lrwxrwxrwx 1 root root       12 3月  27 19:40 /usr/lib/x86_64-linux-gnu/libcuda.so -> libcuda.so.1
lrwxrwxrwx 1 root root       21 3月  27 19:40 /usr/lib/x86_64-linux-gnu/libcuda.so.1 -> libcuda.so.525.105.17
-rwxr-xr-x 1 root root 20988000 4月  27 10:31 /usr/lib/x86_64-linux-gnu/libcuda.so.515.65.01*
-rw-r--r-- 1 root root 29867944 3月  27 19:40 /usr/lib/x86_64-linux-gnu/libcuda.so.525.105.17
```

将 libcuda.so.515.65.01 复制到 lib64，报错

驱动和 library 不一致

```shell
(chat) root@ubuntu-virtual-machine:/data1/chat_model# nvidia smi
Failed to initialize NVML: Driver/library version mismatch
```

https://www.nvidia.com/Download/Find.aspx?lang=en-us

https://www.nvidia.com/download/driverResults.aspx/191320/en-us/

根据这个找到库对应的 515.65.01 版本

下载 sh NVIDIA-Linux-x86_64-515.65.01.run 重新安装即可

```shell
(base) root@ubuntu-virtual-machine:/usr/lib/x86_64-linux-gnu# ll libcuda*
lrwxrwxrwx 1 root root       29 3月  27 19:40 libcudadebugger.so.1 -> libcudadebugger.so.525.105.17
-rw-r--r-- 1 root root 10490248 3月  27 19:40 libcudadebugger.so.525.105.17
lrwxrwxrwx 1 root root       12 4月  28 09:03 libcuda.so -> libcuda.so.1*
lrwxrwxrwx 1 root root       20 4月  28 09:03 libcuda.so.1 -> libcuda.so.515.65.01*
-rwxr-xr-x 1 root root 20988000 4月  28 09:03 libcuda.so.515.65.01*
```



## 5. 运行 test.py(GPU)

修改 test.py 如下

```python
print('----------------------------------------------')

from transformers import AutoTokenizer, AutoModel
import time

print('Tokenizer:')
tokenizer = AutoTokenizer.from_pretrained("chatglm-6b-int4", trust_remote_code=True, revision='int4')
print('Model:')
start_time = time.time()
model = AutoModel.from_pretrained("chatglm-6b-int4", trust_remote_code=True, revision='int4')
print('Time:',time.time()-start_time)
print('Half:')
start_time = time.time()
model = model.half()
print('Time:',time.time()-start_time)
print('Cuda:')
start_time = time.time()
model = model.cuda()
print('Time:',time.time()-start_time)
print('Cuda loaded.')

qlist = ['你好！', '请介绍一下你自己。', '你有哪些能力？', '请从整体介绍一下环保行
业。', '请写一段使用python进行PCA降维可视化的代码。']

history = []
for query in qlist:
    print(query)
    start_time = time.time()
    response, history = model.chat(tokenizer, query, history=history)
    print(response)
    print('Time:',time.time()-start_time)
```

Model：10-15s

Half：0.36s

Cuda：4-15s

````shell
(chat) root@ubuntu-virtual-machine:/data1/chat_model# python test.py
----------------------------------------------
Tokenizer:
Model:
Begin model=cls()
nvrtc : /usr/local/cuda/lib64/libnvrtc.so
cuda : /usr/local/cuda/lib64/libcuda.so
cudart : /usr/local/cuda/lib64/libcudart.so
cublasLt : /usr/local/cuda/lib64/libcublasLt.so
Cannot load cpu kernel, don't use quantized model on cpu.
Using quantization cache
Applying quantization to glm layers
End model=cls()
Load from pt
Loaded from pt
Model evaled
Time: 10.415313243865967
Half:
Time: 0.36229372024536133
Cuda:
Time: 11.275478839874268
Cuda loaded.
你好！
The dtype of attention mask (torch.int64) is not bool
你好👋！我是人工智能助手 ChatGLM-6B，很高兴见到你，欢迎问我任何问题。
Time: 11.327528953552246
请介绍一下你自己。
我可以回答关于我自己的一些问题，例如我的训练数据集，我的算法模型和运行方式等等。同时，我也可以提供一些关于我的主人 ChatGLM-6B 的信息，如果对她感兴趣的话。
Time: 9.367414236068726
你有哪些能力？
作为一个人工智能助手，我具有以下能力：

1. 自然语言处理：我可以理解和生成自然语言，包括文本和语音。

2. 知识库：我可以访问和学习大量的文本和数据集，以便回答用户的问题。

3. 语言生成：我可以生成自然语言文本，例如文章、回复和对话。

4. 自动摘要：我可以自动提取文本中的重点和摘要，以便用户可以快速了解文本内容。

5. 对话：我可以与用户进行对话，回答他们的问题或执行其他任务，例如提供建议或执行自动化任务。

希望这些概括可以提供帮助！
Time: 28.96828293800354
请从整体介绍一下环保行业。
环保行业是指专门从事环境保护和清洁发展的行业，旨在通过解决环境问题来改善环境质量，提高可持续发展水平。环保行业涉及多个领域，包括环保技术、环境监测、环境治理、环保材料、清洁服务、环保法规等。

在环保行业中，环保技术是指利用各种技术和方法来减少环境污染和处理污染物的方法。环境监测是指对环境污染和水质进行检测和分析，以便识别污染物并采取相应的措施。环境治理是指对环境污染进行治理，包括对污染物进行回收、处理和转化。

环保材料是指用于环保领域的材料，这些材料可以用于制造环保产品，减少对环境的污染。清洁服务是指提供清洁服务的公司，包括家庭清洁、商业清洁和工业清洁等。

环保法规是指有关环境保护的法律和政策，旨在保护环境和促进可持续发展。这些法规旨在减少环境污染、保护生态平衡、加强环境监管和推动环保技术的应用。

环保行业是一个涉及众多领域，旨在保护环境和提高可持续发展水平的行业。在日益重视环境保护的当今世界，环保行业的前景看好，同时也面临着巨大的挑战和机遇。
Time: 46.70101070404053
请写一段使用python进行PCA降维可视化的代码。
以下是使用Python进行PCA降维可视化的代码：

```python
import numpy as np
import matplotlib.pyplot as plt

# 定义数据
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 使用PCA进行降维
pca = np.random.randn(data.shape[0], data.shape[1]+1)
pca[:,1] = data

# 打印PCA结果
print("PCA结果：")
print(pca.shape)

# 可视化PCA结果
data_pca = pca.reshape((data.shape[0], data.shape[1]+1))
plt.imshow(data_pca, cmap='gray', aspect='auto', extent=[0, 10, 0, 10])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Pca 降维结果')
plt.show()
```

运行上述代码，将数据经过PCA降维后存储在变量 `data_pca` 中，然后使用 `imshow` 函数将降维后的数据和PCA结果可视化。
Time: 60.72647047042847
````

## 6. 运行情况

| 模型 | 内存(G)   | 显存(G) | 加载(s) | 短句(s) | 长段(s) |
| ---- | --------- | ------- | ------- | ------- | ------- |
| int4 | 6.5/8.5   | 5       | 15-20   | 3-10    | 30-60   |
| int8 | 14.5      | 7-9     | 100-120 | 10-20   | 30-100  |
| 6b   | 13.6/14.5 | 12-13   | 140-250 | 3-10    | 10-40   |

## 7. 报错提示问题

### 6.1 revision 报错

这个是没有提供模型版本号的问题

加入 `revision=` 参数解决，版本号名称可以自定

```python
tokenizer = AutoTokenizer.from_pretrained("chatglm-6b-int4", trust_remote_code=True, revision='int4')
model = AutoModel.from_pretrained("chatglm-6b-int4", trust_remote_code=True, revision='int4')
```

### 6.2 No compiled kernel found

在模型加载阶段，首先会加载一个量化用的 kernel

模型加载到内存后使用 kernel 进行量化，再加载到 GPU 中



chatglm-6b-int4/quantization.py 的量化部分

kwargs 为空，load_cpu_kernel 没有参数传进来

```python
# 441
def quantize(model, weight_bit_width, use_quantization_cache=False, empty_init=False, **kwargs):
    """Replace fp16 linear with quantized linear"""

    query_key_value_quantization_cache = None
    dense_quantization_cache = None
    dense_h_to_4h_quantization_cache = None
    dense_4h_to_h_quantization_cache = None

    try:
        # 修改这一句
        # load_cpu_kernel(**kwargs)
        load_cpu_kernel(kernel_file='quantization_kernels_parallel.so', **kwargs)
    except:
        if kernels is None:  # CUDA kernels failed
            print("Cannot load cpu or cuda kernel, quantization failed:")
            assert kernels is not None
        print("Cannot load cpu kernel, don't use quantized model on cpu.")
```

导致 kernel_file 为空

```python
class CPUKernel:
    def __init__(self, kernel_file="", source_code=default_cpu_kernel_code_path, compile_parallel_kernel=None, parallel_num=None):
        self.load =False
        self.int8WeightExtractionFloat = None
        self.int4WeightExtractionFloat = None
        self.int4WeightCompression = None
        self.SetNumThreads = lambda x: x
# 124
        if compile_parallel_kernel is None:
            compile_parallel_kernel = bool(int(os.cpu_count()) >= 4)

        if compile_parallel_kernel and source_code == default_cpu_kernel_code_path:
            source_code = default_cpu_parallel_kernel_code_path

        kernels = None
        
		if (not kernel_file) or (not os.path.exists(kernel_file)):
            print("No compiled kernel found.")
        	try:
                	# 开始编译 kernel
```



待使用的例子：

我正在编写一个 Python 程序，使用 text2vec 来进行文本向量化。我的程序如下：

```
# Text Embedding Program
from text2vec import SentenceModel
model = SentenceModel(r'D:\Projects\text2vec-base-chinese')
sentences = ['如何更换花呗绑定银行卡', '花呗更改绑定银行卡', '如何办理银行卡', '哈里路大旋风']
embeddings = model.encode(sentences)
print(embeddings)
```
根据模型的编码，我得到了一个名为 embeddings 的数组。embeddings 数组包含4个向量，每个向量对应一个句子，每个向量是一个浮点数的 list
embeddings 的形式类似于：
print(embeddings)

[[-4.4390789e-04 -2.9734781e-01  8.5790151e-01 ... -5.2770150e-01
  -1.4315721e-01 -1.0007849e-01]
 [ 6.5362066e-01 -7.6667674e-02  9.5962387e-01 ... -6.0122484e-01
  -1.6792282e-03  2.1457718e-01]
 [ 3.3719593e-01 -7.9632360e-01  3.8019526e-01 ... -1.9252342e-01
   7.5160302e-02 -1.6093762e-01]
 [-8.8900603e-02 -2.4561442e-02  4.1167963e-01 ...  1.2363576e+00
  -3.4643823e-01 -7.1788603e-01]]
其中 ... 代表省略了中间的维度

现在，我想要使用可视化的方法展示 embeddings 中4个向量的差别，这同时也意味着4个句子的差别。请帮我编写一个python程序来实现可视化 

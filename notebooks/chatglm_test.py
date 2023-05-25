from transformers import AutoTokenizer, AutoModel
import time

print('Loading...')
start_time = time.time()
tokenizer = AutoTokenizer.from_pretrained("chatglm-6b-int4", trust_remote_code=True, revision='int4')
model = AutoModel.from_pretrained("chatglm-6b-int4", trust_remote_code=True, revision='int4').half().cuda()
print('Time:',time.time()-start_time)

qlist = ['你好！', '请介绍一下你自己。', '你有哪些能力？', '请从整体介绍一下环保行业。', '请写一段使用python进行PCA降维可视化的代码。']

history = []
for query in qlist:
    print(query)
    start_time = time.time()
    response, history = model.chat(tokenizer, query, history=history)
    print(response)
    print('Time:',time.time()-start_time)

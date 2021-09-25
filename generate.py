import paddlehub as hub
 
#导入预训练模型
 
module = hub.Module(name="ernie_gen_lover_words")
 
#准备输入开头数据
  
test_texts = ['我','喜欢','你','小倩']
  
#执行文本生成
  
results = module.generate(texts=test_texts, use_gpu=False, beam_width=5)
  
#打印输出结果
  
for result in results:
  
    print(result)
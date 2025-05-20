import jieba

with open('./data/douluodalu.txt', errors='ignore', encoding='utf-8') as fp:
   lines = fp.readlines()
   for line in lines:
       seg_list = jieba.cut(line)
       with open('./data/seg_douluodalu.txt', 'a', encoding='utf-8') as ff:
           ff.write(' '.join(seg_list))
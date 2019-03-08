# coding: utf-8

# 随机从5000个测试数据中挑选100条进行人工评测

import numpy as np
from sumeval.metrics.rouge import RougeCalculator

data_dir = './sorted/'
out_f = 'eval.txt'
order_f = 'order.txt'
max_len = 150  # 太长的review评测太费时间，所以去掉
sample_num = 100
seed = 1
np.random.seed(seed)
models = ['attr', 'attr1', 'attr2', 'hssc', 'seq2seq', 'seq2seqAttn', 'seq2seqCopy']
data = []
reviews = []
refs = []
sums = []


def main():
    for model in models:
        f = open(data_dir + model + '.txt', 'r')
        content = f.read().strip().split('\n\n')
        content = [d.strip().split('\n') for d in content]
        data.append(content)
        f.close()
    data_size = len(data[0])
    for i in range(data_size):
        review = data[0][i][0]
        if len(review.strip().split()) > max_len:
            continue
        reviews.append(review)
        refs.append(data[0][i][1])
        cur_sums = []
        for j in range(len(models)):
            cur_sums.append(data[j][i][2])
        sums.append(cur_sums)
    selected = np.random.choice(range(len(reviews)), sample_num, replace=False)
    np.random.seed(2333)
    model_idx = np.random.permutation(range(len(models)))
    f = open(order_f, 'w')
    for i in model_idx:
        f.write(models[i] + ' ')
    f.close()
    rouge = [.0 for _ in models]
    r = RougeCalculator(stopwords=False, lang="en")
    f = open(out_f, 'w')
    for j, idx in enumerate(selected):
        f.write(str(j+1) + '\r\n')
        f.write(reviews[idx] + '\r\n')
        f.write(refs[idx] + '\n')
        for i in model_idx:
            f.write(sums[idx][i] + '\r\n')
            rouge[i] += r.rouge_n(sums[idx][i], refs[idx], n=1)
        f.write('\r\n')
    f.close()
    rouge = [s / sample_num for s in rouge]
    print(rouge)


if __name__ == '__main__':
    main()

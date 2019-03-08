# coding: utf-8

# 统计结果，得到每个模型平均分数

with open('order.txt', 'r') as f:
    models = f.readline().strip().split()
scores = {}
for model in models:
    scores[model] = .0

cnt = 0
with open('result1.txt', 'r') as f:
    for line in f.readlines():
        if len(line) < 5:
            continue
        cnt += 1
        cur_scores = line.strip().split()
        for i in range(len(models)):
            scores[models[i]] += float(cur_scores[i])

for model in models:
    scores[model] /= cnt

print(scores)

# coding: utf-8

# 将各模型输出的结果进行排序


models = ['attr', 'attr1', 'attr2', 'hssc', 'seq2seq', 'seq2seqAttn', 'seq2seqCopy']


def main():
    for model in models:
        print('Processing %s' % models)
        f = open('./raw/' + model + '.txt', 'r')
        data = []
        for sample in f.read().strip().split('\n\n')
            sample = sample.strip().split('\n')
            assert len(sample) == 3
            data.append(sample)
        data.sort(key=lambda p: p[0])
        f.close()
        f = open('./sorted/' + model + '.txt', 'w')
        for sample in data:
            f.write(sample[0] + '\n')
            f.write(sample[1] + '\n')
            f.write(sample[2] + '\n\n')
        f.close()


if __name__ == '__main__':
    main()

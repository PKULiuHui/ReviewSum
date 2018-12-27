# coding: utf-8

# 封装了一个计算标准Rouge值的函数，可以根据(hyp, ref)计算各种Rouge值
# 鉴于rouge包和标准Rouge结果差距比较大，之后会使用标准Rouge替换

import os
from Rouge155 import Rouge155
import sys
import random

reload(sys)
sys.setdefaultencoding('utf-8')


def get_rouge_score(hyp, ref):
    score = {}
    tmp_dir_name = random.random()
    hyp_dir = './%s_1/' % tmp_dir_name
    ref_dir = './%s_2/' % tmp_dir_name
    if os.path.exists(hyp_dir):
        os.system('rm -r %s' % hyp_dir)
    if os.path.exists(ref_dir):
        os.system('rm -r %s' % ref_dir)
    os.mkdir(hyp_dir)
    os.mkdir(ref_dir)
    with open(os.path.join(hyp_dir, '1.txt'), 'w') as f:
        f.write(hyp)
    with open(os.path.join(ref_dir, '1.txt'), 'w') as f:
        f.write(ref)
    r = Rouge155()
    r.system_dir = hyp_dir
    r.model_dir = ref_dir
    r.system_filename_pattern = '(\d+).txt'
    r.model_filename_pattern = '#ID#.txt'

    output = r.convert_and_evaluate()
    metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'ROUGE-SU4']
    for m in metrics:
        score[m] = {}
        for line in output.split('\n'):
            if m in line:
                if 'Average_R' in line:
                    score[m]['r'] = float(line.split()[3])
                if 'Average_P' in line:
                    score[m]['p'] = float(line.split()[3])
                if 'Average_F' in line:
                    score[m]['f'] = float(line.split()[3])
    if os.path.exists(hyp_dir):
        os.system('rm -r %s' % hyp_dir)
    if os.path.exists(ref_dir):
        os.system('rm -r %s' % ref_dir)
    return score


if __name__ == '__main__':
    s = get_rouge_score('I went to the Mars from my living town.', 'I went to Mars')
    print(s)

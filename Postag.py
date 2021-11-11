"""
词性标注函数部分
"""
import os
from typing import Callable, Any
from Tokenizer import gt
import re
from pos_state import P as pos_state
from pos_start import P as pos_start
from pos_trans import P as pos_trans
from pos_emit import P as pos_emit

_get_abs_path: Callable[[Any], Any] = lambda path: os.path.normpath(os.path.join(os.getcwd(), path))

DEFAULT_DICT = _get_abs_path("dic.txt")

re_han = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\._]+)")  # 中文正则表达式（考虑到一些外来词T恤）
re_skip = re.compile("(\r\n|\s)")  # 换行正则表达式
re_eng = re.compile("[a-zA-Z0-9]+")  # 数字字母
re_num = re.compile("[\.0-9]+")  # 数字
re_han_d = re.compile("([\u4E00-\u9FD5]+)")
re_skip_d = re.compile("([\.0-9]+|[a-zA-Z0-9]+)")
MIN_FLOAT = -3.14e100
MIN_INF = float("-inf")

word_tag_tab = {}
with open('dic.txt', 'rb') as f:
    for line in f:
        line = line.strip().decode()
        if line == "":
            continue
        word, _, tag = line.split(' ')
        word_tag_tab[word] = tag


class pair(object):
    def __init__(self, word, flag):
        self.word = word
        self.flag = flag

    def __unicode__(self):
        return '%s/%s' % (self.word, self.flag)

    def __repr__(self):
        return 'pair(%r, %r)' % (self.word, self.flag)

    def __str__(self):
        return self.__unicode__()

    def __iter__(self):
        return iter((self.word, self.flag))

    def __eq__(self, other):
        return self.word == other.word and self.flag == other.flag


# viterbi算法
def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    mem_path = [{}]
    # 根据状态转移矩阵，获取所有可能的状态
    all_states = trans_p.keys()
    # 时刻t=0，初始状态
    for y in states.get(obs[0], all_states):  # init
        V[0][y] = start_p[y] + emit_p[y].get(obs[0], MIN_FLOAT)
        mem_path[0][y] = ''
    # 时刻t=1,...,len(obs) - 1
    for t in range(1, len(obs)):
        V.append({})
        mem_path.append({})
        # prev_states = get_top_states(V[t-1])
        # 获取前一时刻所有的状态集合
        prev_states = [
            x for x in mem_path[t - 1].keys() if len(trans_p[x]) > 0]

        # 根据前一时刻的状态和状态转移矩阵，提前计算当前时刻的状态集合
        prev_states_expect_next = set(
            (y for x in prev_states for y in trans_p[x].keys()))

        # 根据当前的观察值获得当前时刻的可能状态集合，再与上一步骤计算的状态集合取交集
        obs_states = set(
            states.get(obs[t], all_states)) & prev_states_expect_next

        # 如果当前状态的交集集合为空
        if not obs_states:
            # 如果提前计算当前时刻的状态集合不为空，则当前时刻的状态集合为提前计算当前时刻的状态集合，否则为全部可能的状态集合
            obs_states = prev_states_expect_next if prev_states_expect_next else all_states

        # 当前时刻所处的各种可能的状态集合
        for y in obs_states:
            # 分别获取上一时刻的状态的概率对数，该状态到本时刻的状态的转移概率对数，本时刻的状态的发射概率对数
            # prev_states是当前时刻的状态所对应上一时刻可能的状态集合
            prob, state = max((V[t - 1][y0] + trans_p[y0].get(y, MIN_INF) +
                               emit_p[y].get(obs[t], MIN_FLOAT), y0) for y0 in prev_states)
            V[t][y] = prob
            mem_path[t][y] = state

    # 最后一个时刻
    last = [(V[-1][y], y) for y in mem_path[-1].keys()]
    # if len(last)==0:
    #     print obs
    prob, state = max(last)

    # 从时刻t = len(obs) - 1,...,0，依次将最大概率对应的状态保存在列表中
    route = [None] * len(obs)
    i = len(obs) - 1
    while i >= 0:
        route[i] = state
        state = mem_path[i][state]
        i -= 1
    # 返回最大概率及各个时刻的状态
    return prob, route


# HMM模型预测未登录词
def pos_hmm(sentence):
    prob, pos_list = viterbi(sentence, pos_state, pos_start, pos_trans, pos_emit)
    begin, next = 0, 0

    for i, char in enumerate(sentence):
        pos = pos_list[i][0]
        if pos == 'B':
            begin = i
        elif pos == 'E':
            yield pair(sentence[begin:i + 1], pos_list[i][1])
            next = i + 1
        elif pos == 'S':
            yield pair(char, pos_list[i][1])
            next = i + 1
    if next < len(sentence):
        yield pair(sentence[next:], pos_list[next][1])


def cut_detail(sentence):
    blocks = re_han_d.split(sentence)
    for blk in blocks:
        if re_han_d.match(blk):
            for word in pos_hmm(blk):
                yield word
        else:
            tmp = re_skip_d.split(blk)
            for x in tmp:
                if x:

                    if re_num.match(x):
                        yield pair(x, 'm')
                    elif re_eng.match(x):
                        yield pair(x, 'eng')
                    else:
                        yield pair(x, 'x')


# 基于词典的分词方法
def cut_DAG(sentence):
    DAG = gt.get_DAG(sentence)
    route = {}
    gt.calc(sentence, DAG, route=route)
    x = 0
    buf = ''
    N = len(sentence)
    while x < N:
        y = route[x][1] + 1
        l_word = sentence[x:y]
        if y - x == 1:
            buf += l_word
        else:
            if len(buf) > 0:
                if len(buf) == 1:
                    yield pair(buf, word_tag_tab.get(buf, 'x'))
                    buf = ''
                else:
                    recognized = cut_detail(buf)
                    # recognized = pos_hmm(buf)
                    for t in recognized:
                        yield t
                    buf = ''
            yield pair(l_word, word_tag_tab.get(l_word, 'x'))
        x = y

    if len(buf) > 0:
        if len(buf) == 1: # 单个词
            yield pair(buf, word_tag_tab.get(buf, 'x'))
        else:
            recognized = cut_detail(buf)
            # recognized = pos_hmm(buf)
            for t in recognized:
                yield t


# 带有词性标注的分词主函数
'''
input: sentence
output: (word,flag) (词，词性)对
'''


def cut_main(sentence):
    blocks = re_han.split(sentence)  # 先使用正则表达式划分成单句
    for blk in blocks:
        if re_han.match(blk):
            for word in cut_DAG(blk):  # 在对单句进行分词以及词性标注
                yield word
        else:
            tmp = re_skip.split(blk)
            for x in tmp:
                if x:
                    if re_num.match(x):  # 数字
                        yield pair(x, 'm')
                    elif re_eng.match(x):  # 英文
                        yield pair(x, 'eng')
                    else:  # 未知
                        yield pair(x, 'x')


def cut_pos(sentence):
    ans = cut_main(sentence)
    list = [a.__str__() for a in ans]

    print(list)


if __name__ == '__main__':
    # s = input("请输入句子\n")
    s = "法国启蒙思想家孟德斯鸠曾说过：“一切有权力的人都容易滥用" \
        "权力，这是一条千古不变的经验。有权力的人直到把权力用到" \
        "极限方可休止。”另一法国启蒙思想家卢梭从社会契约论的观点" \
        "出发，认为国家权力是公民让渡其全部“自然权利”而获得的，" \
        "他在其名著《社会契约论》中写道：“任何国家权力无不是以民" \
        "众的权力（权利）让渡与公众认可作为前提的”。"
    # "北京理工大学生前来应聘" \
    # "小明硕士毕业于中国科学院计算所" \
    # "我爱北京天安门" \
    # "武汉市长江大桥" \
    # "1991年 我在睡大觉"

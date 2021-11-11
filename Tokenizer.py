import os
import marshal
import re
from math import log

_get_abs_path = lambda path: os.path.normpath(os.path.join(os.getcwd(), path))

DEFAULT_DICT = _get_abs_path("dic.txt")
re_eng = re.compile('[a-zA-Z0-9]')


class Tokenizer(object):
    def __init__(self, dictionary=DEFAULT_DICT):
        self.dictionary = _get_abs_path(dictionary)
        self.FREQ = {}
        self.total = 0
        self.initialized = False
        self.cache_file = None

    # 获得前缀词典 pfdict
    def gen_pfdict(self, f_name):
        lfreq = {}  # 字典存储  词条:出现次数
        ltotal = 0  # 所有词条的总的出现次数
        with open(f_name, 'rb') as f:  # 打开文件 dic.txt
            for lineno, line in enumerate(f, 1):  # 行号,行
                try:
                    line = line.strip().decode('utf-8')  # 解码为Unicode
                    word, freq, _ = line.split(' ')  # 获得词条 及其出现次数
                    freq = int(freq)
                    lfreq[word] = freq
                    ltotal += freq
                    for ch in range(len(word)):  # 处理word的前缀
                        wfrag = word[:ch + 1]
                        if wfrag not in lfreq:  # word前缀不在lfreq则其出现频次置0
                            lfreq[wfrag] = 0
                except ValueError:
                    raise ValueError(
                        'invalid dictionary entry in %s at Line %s: %s' % (f_name, lineno, line))
        return lfreq, ltotal

    def check_initialized(self):
        if not self.initialized:
            abs_path = _get_abs_path(self.dictionary)
            if self.cache_file:
                cache_file = self.cache_file
            # 默认的cachefile
            elif abs_path:
                cache_file = "seg_mark.cache"

            load_from_cache_fail = True
            # cachefile 存在
            if os.path.isfile(cache_file):

                try:
                    with open(cache_file, 'rb') as cf:
                        self.FREQ, self.total = marshal.load(cf)
                    load_from_cache_fail = False
                except Exception:
                    load_from_cache_fail = True
            if load_from_cache_fail:
                self.FREQ, self.total = self.gen_pfdict(abs_path)
                # 把dict前缀集合,总词频写入文件
                try:
                    with open(cache_file, 'w') as temp_cache_file:
                        marshal.dump((self.FREQ, self.total), temp_cache_file)
                except Exception:
                    # continue
                    pass
            # 标记初始化成功
            self.initialized = True

    # 生成有向无环图
    def get_DAG(self, sentence):
        # 初始化
        self.check_initialized()
        DAG = {}
        N = len(sentence)
        for k in range(N):
            tmplist = []
            i = k
            frag = sentence[k]
            while i < N and frag in self.FREQ:
                if self.FREQ[frag]:
                    tmplist.append(i)
                i += 1
                frag = sentence[k:i + 1]
            if not tmplist:
                tmplist.append(k)
            DAG[k] = tmplist
        return DAG

    # 动态规划，计算最大概率的切分组合
    def calc(self, sentence, DAG, route):
        N = len(sentence)
        route[N] = (0, 0)
        # 对概率值取对数之后的结果(可以让概率相乘的计算变成对数相加,防止相乘造成下溢)
        logtotal = log(self.total)
        # 从后往前遍历句子 反向计算最大概率
        for idx in range(N - 1, -1, -1):
            # 列表推倒求最大概率对数路径
            # route[idx] = max([ (概率对数，词语末字位置) for x in DAG[idx] ])
            # 以idx:(概率对数最大值，词语末字位置)键值对形式保存在route中
            # route[x+1][0] 表示 词路径[x+1,N-1]的最大概率对数,
            # [x+1][0]即表示取句子x+1位置对应元组(概率对数，词语末字位置)的概率对数
            route[idx] = max((log(self.FREQ.get(sentence[idx:x + 1]) or 1) -
                              logtotal + route[x + 1][0], x) for x in DAG[idx])

    # DAG中是以{key:list,...}的字典结构存储
    # key是字的开始位置

    def cut_DAG(self, sentence):
        DAG = self.get_DAG(sentence)
        route = {}
        self.calc(sentence, DAG, route)
        x = 0
        N = len(sentence)
        buf = ''
        while x < N:
            y = route[x][1] + 1
            l_word = sentence[x:y]  # 得到以x位置起点的最大概率切分词语
            if re_eng.match(l_word) and len(l_word) == 1:  # 数字,字母
                buf += l_word
                x = y
            else:
                if buf:
                    yield buf
                    buf = ''
                yield l_word
                x = y
        if buf:
            yield buf
            buf = ''


# 全局实例

gt = Tokenizer()

# 全局变量

get_FREQ = lambda k, d=None: gt.FREQ.get(k, d)
calc = gt.calc
cut = gt.cut_DAG
get_DAG = gt.get_DAG


def cut(sentence):
    ans = gt.cut_DAG(sentence)
    list = [a.__str__() for a in ans]

    print(list)


if __name__ == '__main__':
    s = "法国启蒙思想家孟德斯鸠曾说过：“一切有权力的人都容易滥用" \
        "权力，这是一条千古不变的经验。有权力的人直到把权力用到" \
        "极限方可休止。”另一法国启蒙思想家卢梭从社会契约论的观点" \
        "出发，认为国家权力是公民让渡其全部“自然权利”而获得的，" \
        "他在其名著《社会契约论》中写道：“任何国家权力无不是以民" \
        "众的权力（权利）让渡与公众认可作为前提的”。"
    for a in gt.cut_DAG(s):
        print(a)

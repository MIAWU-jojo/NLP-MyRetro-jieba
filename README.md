# NLP-jieba

### 一、实验完成情况

- 实现基于词典的分词方法和统计分词方法：两类方法中实现一种即可；
- 对分词结果进行词性标注，也可以在分词的同时进行词性标注；
- 对分词及词性标注结果进行评价，包括4个指标：正确率、召回率、F1值和效率。
- 实现了基于HMM模型的命名实体识别：对句子中出现的人名、地名、机构名进行识别。（词性标注时同时实现）（准确率未做评测使用效果可以参见程序使用说明中词性标注的示例）

### 二、实验环境

- python 3.8
- 联想 Y7000 2019(i7 9750H) 8G内存

### 三、 实验过程

#### 1. 基于统计词典的分词方法

##### 1. 1 算法设计及程序结构

##### 1.1.0 常见的基于词典的分词方法有正向最大匹配、逆向最大匹配、双向最大匹配和最少词数分词。

（1）正向最大匹配

对输入的句子从左至右，取词典中最长单词的个数作为第一次取词的个数，在词典中进行扫描，若不匹配，则逐字递减；若匹配，则取出当前词，从后面的词开始正向最大匹配，组不了词的字单独划开。其分词基本原则是：词的颗粒度越大越好；切分结果中非词典词越少越好；总体词数越少越好。

（2）逆向最大匹配

分词原则与正向最大匹配相同，但顺序不是从首字开始，而是从末字开始，而且它使用的分词词典是逆序词典，其中每个词条都按逆序方式存放。在实际处理时，先将句子进行倒排处理，生成逆序句子，然后根据逆序词典，对逆序句子用正向最大匹配处理。

（3） 双向最大匹配

将正向最大匹配与逆向最大匹配组合起来，对句子使用这两种方式进行扫描切分，如果两种分词方法得到的匹配结果相同，则认为分词正确，否则，按最小集处理。


##### 1.1.1 本文分词使用了前缀词典方法

1. ###### 获得前缀词典`pfdict`

```python
    # 获得前缀词典 pfdict
    def get_pfdict(self, f_name):
        ffreq = {}  # 字典存储  词条:出现次数
        ftotal = 0  # 所有词条的总的出现次数
        with open(f_name, 'rb') as f:  # 打开文件 dic.txt
            for lineno, line in enumerate(f, 1):  # 行号,行
                line = line.strip().decode('utf-8')  # 解码为Unicode
                word, freq, _ = line.split(' ')  # 获得词条 及其出现次数                         
                ffreq[word] = int(freq)
                ftotal += freq
                for ch in range(len(word)):  # 处理word的前缀                
                	wfrag = word[:ch + 1]
                    if wfrag not in lfreq:  # word前缀不在lfreq则其出现频次置0
                        ffreq[wfrag] = 0
        return ffreq, ftotal
```

2. ###### 依据`pfdict`构建句子的DAG有向无环图

```python
    # 生成有向无环图
    def get_DAG(self, sentence):
        # 初始化
        self.check_initialized()
        DAG = {}  # dict{key:list[i,j,k····]} 开始位置为key的可能结束位置
        N = len(sentence)
        for k in range(N):
            tmplist = []
            i = k
            frag = sentence[k]
            while i < N and frag in self.FREQ:  # 如果词在前缀词典中
                if self.FREQ[frag]:
                    tmplist.append(i)
                i += 1
                frag = sentence[k:i + 1]
            if not tmplist:
                tmplist.append(k)
            DAG[k] = tmplist
        return DAG
```

3. ###### 采用动态规划方法计算最大概率的路径

   从后向前遍历`sentence`计算最大概率将路径记录在`route`


```python
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
            route[idx] = max((log(self.FREQ.get(sentence[idx:x + 1]) or 1) -logtotal + route[x + 1][0], x) for x in DAG[idx])
```

4. ###### 回溯获得最终分词路径模块

```python
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
            if re_eng.match(l_word) and len(l_word) == 1:  # 数字,字母 ,re_eng是数字字母的正则表达式
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
```

##### 1.2 效果评测

我们从正确率、召回率、F1值和效率四个维度对分词和词性标注功能进行评价

以`jieba`库作为评测标准，评测文本为`test.txt` （《围城》文章 ）

使用`test.py`进行评测

| 正确率 | 召回率 | F1值   | 效率        |
| ------ | ------ | ------ | ----------- |
| 92.10% | 97.42% | 94.69% | 339.748KB/s |

#### 2. 基于HMM与Viterbi算法的词性标注

##### 2.1 算法设计及程序结构

总体算法设计：

先使用正则表达式进行初步划分，若是数字字母或符号标注为其他类型
若是汉字先使用基于DAG的分词，如果已登录的词就使用词典分词并标识，未登录词使用3-gram隐马尔可夫模型+维特比算法进行词性预测,可以识别人名机构名。



1. ###### HMM模型和Viterbi算法



**HMM模型作的两个基本假设：**

- 齐次马尔科夫性假设，即假设隐藏的马尔科夫链在任意时刻t的状态只依赖于其前一时刻的状态，与其它时刻的状态及观测无关，也与时刻t无关；

  $P(states[t] | states[t-1],observed[t-1],...,states[1],observed[1]) = P(states[t] | states[t-1]) t = 1,2,...,T$

- 观测独立性假设，即假设任意时刻的观测只依赖于该时刻的马尔科夫链的状态，与其它观测和状态无关，

  $P(observed[t] | states[T],observed[T],...,states[1],observed[1]) = P(observed[t] | states[t]) t = 1,2,...,T$

**HMM模型中的五元组表示：**

- 观测序列；

- 隐藏状态序列；

- 状态初始概率；

- 状态转移概率；

- 状态发射概率；

词性标注问题即为预测问题，也称为解码问题，已知模型 $λ=(A,B,π)$ 和观测序列 $O=(o_1,o_2,...,o_T)$ ，求对给定观测序列条件概率$ P(S|O)$ 最大的状态序列 $I=(s_1,s_2,...,s_T)$ ，即给定观测序列。求最有可能的对应的词性标注序列；

使用**BMES标注方法**，BMES标注方法是用“B、M、E、S”四种标签对词语中不同位置的字符进行标注，B表示一个词的词首位置，M表示一个词的中间位置，E表示一个词的末尾位置，S表示一个单独的字

例如对于”大玩学城“这个句子隐藏序列为[(u'S', u'a'), (u'B', u'n'), (u'E', u'n'), (u'B', u'n')分词]分词结果为[S/BE/B]

**Viterbi函数**获得模型参数即可预测概率最大的转移序列

先计算各个初始状态的对数概率值，然后递推计算，每时刻某状态的对数概率值取决于上一时刻的对数概率值、上一时刻的状态到这一时刻的状态的转移概率`trans_p[i in prev_states]`、这一时刻状态转移到当前的字的发射概率`emit_p`三部分组成

```python
"""
obs: 观测序列
states: 状态序列
start_p: 状态初始概率
trans_p: 状态转移概率
emit_p: 状态发射概率
"""
MIN_FLOAT = -3.14e100
MIN_INF = float("-inf")


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

```

2. ###### 载入HMM模型

```python
from pos_state import P as pos_state 
from pos_start import P as pos_start
from pos_trans import P as pos_trans
from pos_emit import P as pos_emit 
word_tag_tab = {} # 词性词典
with open('dic.txt', 'rb') as f:
    for line in f:
        line = line.strip().decode()
        if line == "":
            continue
        word, _, tag = line.split(' ')
        word_tag_tab[word] = tag
        
"""
pos_state.py存储了离线统计的字及其对应的状态；
pos_emit.py存储了状态到字的发射概率的对数值；
pos_start.py存储了初始状态的概率的对数值；
pos_trans.py存储了前一时刻的状态到当前时刻的状态的转移概率的对数值；
"""

```

3. ###### 分词函数

```python
# 带有词性标注的分词主函数
# re_han = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\._]+)")
# re_skip = re.compile("(\r\n|\s)")
# re_eng = re.compile("[a-zA-Z0-9]+")
# re_num = re.compile("[\.0-9]+")
'''
input: sentence
output: (word,flag) (词，词性)对
'''
def cut_main(sentence):
    blocks = re_han.split(sentence)# 先使用正则表达式划分成单句
    for blk in blocks:
        if re_han.match(blk):
            for word in cut_DAG(blk): #在对单句进行分词以及词性标注
                yield word
        else:
            tmp = re_skip.split(blk)
            for x in tmp:
                if x:
                    if re_num.match(x): # 数字
                        yield pair(x, 'm')
                    elif re_eng.match(x): # 英文
                        yield pair(x, 'eng')
                    else:		# 未知
                        yield pair(x, 'x')
```

```python
# 基于词典的分词方法
# gt 是 Tokenizer.py 中定义的全局分词器
def cut_DAG(sentence):
    # 构图
    DAG = gt.get_DAG(sentence)
    route = {}
    # 计算最大概率路径
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
                    # 词 词性字典中有该词，则将词性赋予该词
                    yield pair(buf, word_tag_tab.get(buf, 'x'))
                    buf = ''
                else:
                    recognized = cut_detail(buf)
                    for t in recognized:
                        yield t
                    buf = ''
            # 默认是x词性
            yield pair(l_word, word_tag_tab.get(l_word, 'x'))
        x = y

    if len(buf) > 0:
        if len(buf) == 1: # 单个词
            yield pair(buf, word_tag_tab.get(buf, 'x'))
        else:
            recognized = cut_detail(buf)
            for t in recognized:
                yield t
```



```python
# 先将单词划分为较短汉语 加速进行pos_hmm预测# re_han_d = re.compile("([\u4E00-\u9FD5]+)")# re_skip_d = re.compile("([\.0-9]+|[a-zA-Z0-9]+)")def cut_detail(sentence):    blocks = re_han_d.split(sentence)    for blk in blocks:        if re_han_d.match(blk):            for word in pos_hmm(blk):                yield word        else:            tmp = re_skip_d.split(blk)            for x in tmp:                if x:                    if re_num.match(x):                        yield pair(x, 'm')                    elif re_eng.match(x):                        yield pair(x, 'eng')                    else:                        yield pair(x, 'x')# HMM模型预测未登录词def pos_hmm(sentence):    prob, pos_list = viterbi(sentence, pos_state, pos_start, pos_trans, pos_emit)    begin, next = 0, 0	    for i, char in enumerate(sentence):        pos = pos_list[i][0]        if pos == 'B': # 为词头            begin = i        elif pos == 'E': # 为词尾            yield pair(sentence[begin:i + 1], pos_list[i][1])            next = i + 1        elif pos == 'S': # 为单词            yield pair(char, pos_list[i][1])            next = i + 1    if next < len(sentence):        yield pair(sentence[next:], pos_list[next][1])
```



##### 2.2 效果评测

我们从正确率、召回率、F1值和效率四个维度对分词和词性标注功能进行评价

以`jieba`库作为评测标准，评测文本为`test.txt` （《围城》文章 ）

使用`test.py`进行评测

| 正确率 | 召回率 | F1值   | 效率      |
| ------ | ------ | ------ | --------- |
| 96.36% | 92.56% | 94.42% | 49.28KB/s |





#### 3.程序完成过程中所遇问题及解决方法	

1. 问题：HMM 模型中，大部分词语的发射概率较低，随着句子长度的增加（约为120词），路径的概率变得很小，容易被视为0。造成比较不易

   解决方法：选择路径概率取对数，概率相乘转化为对数相加，避免路径概率下溢

2. 对于1999年分词为1  9  9  9 年 的情况 而不是 1999 年 使用了正则匹配方法

3. 词料库的处理先使用了jieba库获得了dic.txt文件以及相应的转移概率文件

4. 为了加速分词过程使用了cache文件提前存好前缀词典

### 四、 程序使用说明

#### 1. 程序功能

- 进行中文语句段落的分词
- 词性标注及未登录词(人名机构名少数实体)识别

#### 2. 词性说明

词性标注参考jieba分词的词性标注方法：

```
- a 形容词  	- ad 副形词  	- ag 形容词性语素  	- an 名形词  - b 区别词  - c 连词  - d 副词  	- df   	- dg 副语素  - e 叹词  - f 方位词  - g 语素  - h 前接成分  - i 成语 - j 简称略称  - k 后接成分  - l 习用语  - m 数词  	- mg 	- mq 数量词  - n 名词  	- ng 名词性语素  	- nr 人名  	- nrfg  古代汉语人名  	- nrt  音译名	- ns 地名  	- nt 机构团体名  	- nz 其他专名  - o 拟声词  - p 介词  - q 量词  - r 代词  	- rg 代词性语素  	- rr 人称代词  	- rz 指示代词  - s 处所词  - t 时间词  	- tg 时语素  - u 助词  	- ud 结构助词 得	- ug 时态助词	- uj 结构助词 的	- ul 时态助词 了	- uv 结构助词 地	- uz 时态助词 着- v 动词  	- vd 副动词	- vg 动词性语素  	- vi 不及物动词  	- vn 名动词  	- vq - x 非语素词（包含标点符号）- y 语气词  - z 状态词  	- zg 
```




#### 3. 使用方法

##### 3.1 分词方法

- 函数`Tokenizer.cut`：
  - 输入：sentence 需要分词和标注词性的句子
  - 输出：list[word] 分完词组成的列表

示例如下：

```python
>>> from Tokenizer import cut>>> cut("我爱自然语言处理")['我', '爱', '自然语言', '处理']# 人名 机构名分词识别>>> cut('江泽民同志毕业于上海交通大学')['江泽民', '同志', '毕业', '于', '上海交通大学']
```



##### 3.2 词性标注方法

- 函数`Postag.cut_pos`：
  - 输入：sentence 需要分词和标注词性的句子
  - 输出：list[word/flag] 分完词与词性的列表

示例如下：

```python
>>> from Postag import cut_pos>>> cut_pos("我爱自然语言处理")['我/r', '爱/v', '自然语言/l', '处理/v']# 人名 机构名分词识别>>> cut_pos("江泽民同志毕业于上海交通大学")['江泽民/nr', '同志/n', '毕业/n', '于/p', '上海交通大学/nt']>>> cut_pos("1943年，时年17岁的江泽民参加地下党领导的学生运动。")['1943/m', '年/m', '，/x', '时/ng', '年/m', '17/m', '岁/m', '的/uj', '江泽民/nr', '参加/v', '地下党/n', '领导/n', '的/uj', '学生/n', '运动/vn', '。/x']>>> cut_pos("北京是中国的首都")['北京/ns', '是/v', '中国/ns', '的/uj', '首都/d']>>> cut_pos("他在学生时代就学会了二胡、笛子等传统乐器，能演奏传统曲目《高山流水》、《春江花月夜》。")['他/r', '在/p', '学生/n', '时代/n', '就/d', '学会/n', '了/ul', '二胡/n', '、/x', '笛子/n', '等/u', '传统/n', '乐器/n', '，/x', '能/v', '演奏/v', '传统/n', '曲目/n', '《/x', '高山流水/ns', '》、《/x', '春江花月夜/nz', '》。/x']
```




import os
import jieba.posseg
from Postag import cut_main
from Tokenizer import gt
from timeit import default_timer as timer

if __name__ == '__main__':

    # 分词标注测评
    ok = 0
    Precision = 0
    Recall = 0
    tic = timer()
    with open('test.txt', 'r', encoding='utf-8') as f:

        while True:
            line = f.readline()
            if not line:
                break
            pred = list(gt.cut_DAG(line))
            # 以jieba分词为评测标准
            ans = list(jieba.cut(line, ))
            # print(ans)
            Precision = Precision + len(pred)
            Recall = Recall + len(ans)
            for a in pred:
                if a in ans:
                    ok = ok + 1
            # print(ok)
    toc = timer()
    with open('test.txt', 'r', encoding='utf-8') as f:

        while True:
            line = f.readline()
            if not line:
                break
            # pred = list(cut(line))
            # 以jieba分词为评测标准
            ans = list(jieba.cut(line, ))
            # print(ans)
            # Precision = Precision + len(pred)
            # Recall = Recall + len(ans)
            # for a in pred:
            #     if a in ans:
            #         ok = ok + 1
            # print(ok)
    tac = timer()
    precision = ok / Precision
    recall = ok / Recall
    f1 = 2 * precision * recall / (precision + recall)
    print("-----------------分词------------------")
    print("Precision:{}, Recall:{}, F1:{}".format(precision, recall, f1))
    fsize = os.path.getsize('test.txt')
    fszie = fsize / 1024
    time = 2 * toc - tic - tac
    speed = fszie / time
    # print(fszie)
    # print(tac - toc)
    # print(toc - tic)
    print("用时 %g s" % (time))
    print("速度为 %g KB/s" % (speed))

    # 词性标注评测
    print("-------------------词性标注+分词--------------------")
    ok = 0
    Precision = 0
    Recall = 0
    tic = timer()
    with open('test.txt', 'r', encoding='utf-8') as f:

        while True:
            line = f.readline()
            if not line:
                break
            pred = list(cut_main(line))
            # 以jieba分词为评测标准
            ans = list(jieba.posseg.cut(line, ))
            # print(ans)
            Precision = Precision + len(pred)
            Recall = Recall + len(ans)
            for a in pred:
                if a in ans:
                    ok = ok + 1
            # print(ok)
    toc = timer()
    with open('test.txt', 'r', encoding='utf-8') as f:

        while True:
            line = f.readline()
            if not line:
                break
            # pred = list(cut(line))
            # 以jieba分词为评测标准
            ans = list(jieba.posseg.cut(line, ))
            # print(ans)
            # Precision = Precision + len(pred)
            # Recall = Recall + len(ans)
            # for a in pred:
            #     if a in ans:
            #         ok = ok + 1
            # print(ok)
    tac = timer()
    precision = ok / Precision
    recall = ok / Recall
    f1 = 2 * precision * recall / (precision + recall)
    print("Precision:{}, Recall:{}, F1:{}".format(precision, recall, f1))
    fsize = os.path.getsize('test.txt')
    fszie = fsize / 1024
    time = 2 * toc - tic - tac
    speed = fszie / time
    # print(fszie)
    # print(tac - toc)
    # print(toc - tic)
    print("用时 %g s" % (time))
    print("速度为 %g KB/s" % (speed))

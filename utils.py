import os
import re
import jieba

def load_stopwords():
    stopwords = [line.strip() for line in open(
        'baidu_stopwords.txt', encoding='UTF-8').readlines()]
    return stopwords

def cut_str(str):
    return jieba.lcut(str.strip(), cut_all=True)

def remove_stopwords(words, stop_words):
    out = []
    for word in words:
        word = remote_illegal_chars(word)
        if len(word) == 0:
            continue
        if word not in stop_words:
            out.append(word)
    return out


def remote_illegal_chars(word):
    return re.sub(r'[0123456789#,:、（，。\\：）？】【\/\(\)\|\[\]\-\.\*]+', '', word).strip()

def cut_and_remove_stopwords(str, stopwords):
    words = cut_str(str)
    words = remove_stopwords(words, stopwords)
    return words

def read_docs():
    folder = "./docs"
    out = []
    for f in os.listdir(folder):
        full_path = os.path.join(folder, f)
        if os.path.isfile(full_path):
            out.append(" ".join(line.strip()
                                for line in open(full_path, encoding="UTF-8").readlines()))
    return out

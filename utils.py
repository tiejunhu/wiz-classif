import os
import re
import jieba


def load_stopwords():
    stopwords = [line.strip() for line in open(
        'stopwords.txt', encoding='UTF-8').readlines()]
    return stopwords


STOP_WORDS = load_stopwords()


def cut_str(str):
    return jieba.lcut(str.strip(), cut_all=False)


def remove_stopwords(words):
    out = []
    for word in words:
        word = remote_illegal_chars(word)
        if len(word) == 0:
            continue
        if len(word) == 1:
            continue
        if word.lower() not in STOP_WORDS:
            out.append(word)
    return out


def remote_illegal_chars(word):
    return re.sub(r'[0123456789#,`:、（，。\\：）？】【「\/\(\)\|\[\]\-\.\*]+', '', word).strip()


def cut_and_remove_stopwords(str):
    words = cut_str(str)
    words = remove_stopwords(words)
    return words


def read_file(full_path):
    if os.path.isfile(full_path):
        return " ".join(line.strip()
                        for line in open(full_path, encoding="UTF-8").readlines())
    raise full_path + "is not file"


def read_docs(proc=False):
    folder = "./docs"
    out = []
    for f in os.listdir(folder):
        full_path = os.path.join(folder, f)
        str = read_file(full_path)
        if proc:
            str = proc(str, f)
        out.append(str)
    return out

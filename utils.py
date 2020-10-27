import os
import re
import hanlp


def load_stopwords():
    stopwords = [line.strip() for line in open(
        'stopwords.txt', encoding='UTF-8').readlines()]
    return stopwords


STOP_WORDS = load_stopwords()

TOKENIZER = hanlp.load('LARGE_ALBERT_BASE')


def cut_str(data):
    print("cutting data")
    words = TOKENIZER(data)
    flat_words = [item for sublist in words for item in sublist]
    return flat_words


def remove_stopwords(words):
    print("removing stopwords")
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


def cut_and_remove_stopwords(data):
    words = cut_str(data)
    words = remove_stopwords(words)
    return words


def read_file(full_path):
    if os.path.isfile(full_path):
        return " ".join(line.strip()
                        for line in open(full_path, encoding="UTF-8").readlines())
    raise full_path + "is not file"


def read_file_into_lines(full_path):
    print("reading " + full_path)
    lines = []
    if os.path.isfile(full_path):
        for line in open(full_path, encoding="UTF-8").readlines():
            line = line.strip()
            if len(line) <= 126:
                lines.append(line)
            else:
                seperator = "。"
                if seperator not in line:
                    seperator = "."
                for ln in line.split(seperator):
                    if len(ln) <= 126:
                        lines.append(ln)
        return lines
    raise full_path + "is not file"


def read_docs(proc=False):
    folder = "./docs"
    out = []
    for f in os.listdir(folder):
        full_path = os.path.join(folder, f)
        data = read_file_into_lines(full_path)
        if proc:
            data = proc(data, f)
        out.append(data)
    return out

import os
import re
import sys
from qqseg import qqseg


def load_stopwords():
    stopwords = [line.strip() for line in open(
        'stopwords.txt', encoding='UTF-8').readlines()]
    return stopwords

def init_qqseg():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if not qqseg.TCInitSeg(os.path.join(dir_path, "qqseg/data")):
        print('Init QQSeg failed from %s' % sys.argv[1], file=sys.stderr)
        sys.exit(1)
    print('Init QQSeg successful')
    return qqseg.TCCreateSegHandle(
        qqseg.TC_CRF | qqseg.TC_PER_W | qqseg.TC_LOC_W | qqseg.TC_ORG_W | 
        qqseg.TC_NER_DL | qqseg.TC_CUS | qqseg.TC_PRODUCTION
    )


STOP_WORDS = load_stopwords()
QQ_HANDLE = init_qqseg()


def cut_str(data):
    print("cutting data")
    words = []
    qqseg.TCSegment(QQ_HANDLE, data, len(data.encode("UTF-8")), qqseg.TC_UTF8)
    for i in range(qqseg.TCGetResultCnt(QQ_HANDLE)):
        token = qqseg.TCGetBasicTokenAt(QQ_HANDLE, i)
        words.append(token.word)
    return words


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
        data = read_file(full_path)
        if proc:
            data = proc(data, f)
        out.append(data)
    return out

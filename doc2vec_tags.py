from gensim.corpora import Dictionary
from gensim.models import TfidfModel

import utils

file_names = []


def process_doc(str, f):
    file_names.append(f)
    return utils.cut_and_remove_stopwords(str)

docs = utils.read_docs(process_doc)
dct = Dictionary(docs)
corpus = [dct.doc2bow(line) for line in docs]
model = TfidfModel(corpus)

tfidf = model[corpus]
for index, doc in enumerate(tfidf):
    words = sorted(doc, key=lambda tup: tup[1], reverse=True)
    print(file_names[index], end=': ')
    for i in range(5):
        print(dct[words[i][0]], end=' ')
    print()
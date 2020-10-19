from gensim import corpora
from gensim.similarities import Similarity
import os
import sys
import os.path

import utils


def read_docs():
    folder = "./docs"
    out = []
    for f in os.listdir(folder):
        full_path = os.path.join(folder, f)
        if os.path.isfile(full_path):
            out.append(" ".join(line.strip()
                                for line in open(full_path, encoding="UTF-8").readlines()))
    return out


stop_words = utils.load_stopwords()

raw_documents = read_docs()

corpora_documents = []
for item_text in raw_documents:
    item_str = utils.cut_and_remove_stopwords(item_text, stop_words)
    corpora_documents.append(item_str)

# 生成字典和向量语料
dictionary = corpora.Dictionary(corpora_documents)
corpus = [dictionary.doc2bow(text) for text in corpora_documents]

similarity = Similarity('Similarity-index', corpus, num_features=2000)

test_data_1 = '你好，我想问一下我想离婚他不想离，孩子他说不要，是六个月就自动生效离婚'
test_cut_raw_1 = utils.cut_and_remove_stopwords(test_data_1, stop_words)
test_corpus_1 = dictionary.doc2bow(test_cut_raw_1)
similarity.num_best = 5
# 返回最相似的样本材料,(index_of_document, similarity) tuples
print(similarity[test_corpus_1])

print('################################')

test_data_2 = '家人因涉嫌运输毒品被抓，她只是去朋友家探望朋友的，结果就被抓了，还在朋友家收出毒品，可家人的身上和行李中都没有。现在已经拘留10多天了，请问会被判刑吗'
test_cut_raw_2 = utils.cut_and_remove_stopwords(test_data_2, stop_words)
test_corpus_2 = dictionary.doc2bow(test_cut_raw_2)
similarity.num_best = 5
# 返回最相似的样本材料,(index_of_document, similarity) tuples
print(similarity[test_corpus_2])

import os
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

import utils


def read_docs_to_tagged_documents():
    folder = "./docs"
    out = []
    for fname in os.listdir(folder):
        full_path = os.path.join(folder, fname)
        if os.path.isfile(full_path):
            fstr = " ".join(line.strip() for line in open(
                full_path, encoding="UTF-8").readlines())
            out.append(TaggedDocument(
                utils.cut_and_remove_stopwords(fstr), [fname]))
    return out


docs = read_docs_to_tagged_documents()

model = Doc2Vec(docs, dm=0, vector_size=128, min_count=0, workers=4, epochs=10)

model.save('model1.model')

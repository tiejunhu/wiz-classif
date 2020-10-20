import os
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

import utils


STOP_WORDS = utils.load_stopwords()


def read_docs_to_tagged_documents():
    folder = "./docs"
    out = []
    for f in os.listdir(folder):
        full_path = os.path.join(folder, f)
        if os.path.isfile(full_path):
            str = " ".join(line.strip() for line in open(
                full_path, encoding="UTF-8").readlines())
            out.append(TaggedDocument(
                utils.cut_and_remove_stopwords(str, STOP_WORDS), [f]))
    return out


docs = read_docs_to_tagged_documents()

model = Doc2Vec(docs, dm=0, vector_size=128, min_count=0, workers=4, epochs=10)

model.save('model1.model')

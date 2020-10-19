import gensim
import os

import utils



model = gensim.models.LdaModel(corpus=corpus,
                               id2word=dictionary,
                               num_topics=50,
                               passes=10)


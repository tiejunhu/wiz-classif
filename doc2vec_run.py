import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

import utils


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def parseArgs():
    parser = argparse.ArgumentParser(description='doc2vec train and clustering.')

    parser.add_argument(
            '--docs', dest='docs_dir', default='./docs', type=dir_path, 
            help="folder that hold all the txt files, default ./docs")
    parser.add_argument(
            '--model', dest='model_file', default='./doc2vec.model',
            help="model file to read/write, default ./doc2vec.model")
    parser.add_argument(
            '--train-only', action='store_true', dest='train_only', default=False, 
            help="only do the training")
    parser.add_argument(
            '--cluster-only', action='store_true', dest='cluster_only', default=False, 
            help="only do the cluster")

    args = parser.parse_args()

    print("process text files from: " + args.docs_dir)
    print("model file: " + args.model_file)

    if args.train_only and args.cluster_only:
        parser.error("--train-only and --cluster-only cannot be set together")

    return args

def train_model(docs_dir, model_file):
    docs = utils.read_docs(lambda str, f: TaggedDocument(utils.cut_and_remove_stopwords(str), [f]))

    model = Doc2Vec(docs, dm=0, vector_size=128, min_count=0, workers=4, epochs=10)

    print("saving model to " + model_file)

    model.save(model_file)


def hover_text(sc, fig, ax, names):
    annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(ind):

        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = "{}".format(" ".join([names[n] for n in ind["ind"]]))
        annot.set_text(text)
        annot.get_bbox_patch().set_alpha(0.4)


    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)


def cluster(model_file):
    model = Doc2Vec.load(model_file)

    cluster_model = AgglomerativeClustering(n_clusters=None, distance_threshold=8)

    cluster_model.fit(model.docvecs.vectors_docs)
    labels = cluster_model.labels_.tolist()

    # doctags(filename) order align with labels
    names = []
    for k in model.docvecs.doctags:
        v = model.docvecs.doctags[k]
        names.insert(v.offset, k)

    pca = PCA(n_components = 2).fit(model.docvecs.vectors_docs)
    datapoint = pca.transform(model.docvecs.vectors_docs)

    cs = plt.get_cmap('rainbow')(np.linspace(0, 1, max(labels) + 1))
    colors = [cs[i] for i in labels]

    fig, ax = plt.subplots()
    sc = plt.scatter(datapoint[:, 0], datapoint[:, 1], c=colors)

    hover_text(sc, fig, ax, names)

    plt.show()


if __name__ == "__main__":
    args = parseArgs()
    if not args.cluster_only:
        print("start training")
        train_model(args.docs_dir, args.model_file)
    if not args.train_only:
        print("clustering")
        cluster(args.model_file)

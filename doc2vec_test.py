from gensim.models import Doc2Vec

model = Doc2Vec.load('model1.model')


def print_most_similar(tag):
    print(tag)
    print(model.docvecs.most_similar(tag))

print_most_similar('01.txt')
print_most_similar('04.txt')
print_most_similar('07.txt')
print_most_similar('10.txt')
print_most_similar('14.txt')

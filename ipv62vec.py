from gensim.models import word2vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
import seaborn as sns
import pandas as pd
import numpy as np
import logging
import ipaddress


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s, level=logging.INFO')

dataset_path = "data/processed_data/word_data.txt"
model_path = 'models/ipv62vec.model'


def read_data():
    f = open(dataset_path, "r")
    data = f.readlines()
    f.close()
    return data


def word2vec_processing(nybbles):
    null = ""
    address = null.join(nybbles) + "\n"

    word2vec_data = []
    location_alpha = '0123456789abcdefghijklmnopqrstuv'
    for nybble, location in zip(address, location_alpha):
        word2vec_data.append(nybble + location)
    return word2vec_data


def train():
    sentences = word2vec.LineSentence(dataset_path)
    model = word2vec.Word2Vec(sentences, alpha=0.025, min_count=5, size=100, window=5,
                              sg=0, hs=0, negative=5, ns_exponent=0.75, iter=5)
    model.save(model_path)


def word_tsne_picture():
    model = word2vec.Word2Vec.load(model_path)

    vocab = list(model.wv.vocab.keys())

    print(model.wv.most_similar('20'))
    X_tsne = TSNE(n_components=2, learning_rate=200, perplexity=30).fit_transform(model.wv[vocab])

    fig = plt.figure(figsize=(8, 4))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=5, edgecolors="black", linewidths=0.3)
    fig.tight_layout()
    plt.savefig("images/word_tsne_ipv62vec_without_text.png", dpi=800)

    fig = plt.figure(figsize=(8, 4))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=5, edgecolors="black", linewidths=0.3)
    for i in range(len(X_tsne)):
        x = X_tsne[i][0]
        y = X_tsne[i][1]
        plt.text(x, y, vocab[i],  size=3)
    fig.tight_layout()
    plt.savefig("images/word_tsne_ipv62vec.png", dpi=800)

    content_word = [str(hex(i))[-1] for i in range(0, 16)]
    situation_word = [str(i) for i in range(0, 10)] + [chr(i) for i in range(97, 119)]
    content_list = []
    situation_list = []
    for i in content_word:
        content = []
        for j in situation_word:
            word = i + j
            if word in vocab:
                content.append(word)
        content_list.append(content)
    for i in situation_word:
        situation = []
        for j in content_word:
            word = j + i
            if word in vocab:
                situation.append(word)
        situation_list.append(situation)

    plt.figure(figsize=(10, 8))
    for content in content_list:
        x = []
        y = []
        for word in content:
            x.append(X_tsne[vocab.index(word), 0])
            y.append(X_tsne[vocab.index(word), 1])
        plt.scatter(x, y, s=5)

    for i in range(len(X_tsne)):
        x = X_tsne[i][0]
        y = X_tsne[i][1]
        plt.text(x, y, vocab[i], size=3)
    plt.savefig("images/word_tsne_ipv62vec_content.png", dpi=300)

    plt.figure(figsize=(10, 8))
    for situation in situation_list:
        x = []
        y = []
        for word in situation:
            x.append(X_tsne[vocab.index(word), 0])
            y.append(X_tsne[vocab.index(word), 1])
        plt.scatter(x, y, s=5)

    for i in range(len(X_tsne)):
        x = X_tsne[i][0]
        y = X_tsne[i][1]
        plt.text(x, y, vocab[i], size=3)
    plt.savefig("images/word_tsne_ipv62vec_situation.png", dpi=300)


def address_tsne_picture():
    model = word2vec.Word2Vec.load(model_path)
    vocab = list(model.wv.vocab.keys())
    X_tsne = TSNE(n_components=2, learning_rate=200, perplexity=30).fit_transform(model.wv[vocab])
    samples = read_data()
    address_vectors = [sample[:-1].split() for sample in samples[0:10000]]
    x = []
    y = []
    for address_vector in address_vectors:
        x_one_sample = []
        y_one_sample = []
        for word in address_vector:
            x_one_sample.append(X_tsne[vocab.index(word), 0])
            y_one_sample.append(X_tsne[vocab.index(word), 1])
        x.append(np.mean(x_one_sample))
        y.append(np.mean(y_one_sample))
    plt.figure(figsize=(10, 8))
    plt.scatter(x, y, s=5)
    plt.savefig("images/address_tsne_ipv62vec.png", dpi=300)

    dim_reduced_data = []
    for i, j in zip(x, y):
        dim_reduced_data.append([i, j])
    dim_reduced_data = pd.DataFrame(dim_reduced_data)
    dim_reduced_data.columns = ['x', 'y']

    cluster(dim_reduced_data)

    return dim_reduced_data


def cluster(data):
    db = DBSCAN(eps=0.1, min_samples=10).fit(data)
    data['labels'] = db.labels_
    labels = db.labels_
    raito = data.loc[data['labels'] == -1].x.count() / data.x.count()
    print('噪声比:', format(raito, '.2%'))
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('分簇的数目: %d' % n_clusters_)
    print("轮廓系数: %0.3f" % metrics.silhouette_score(data, labels))
    # str_labels = [str(label) for label in db.labels_]
    # data['labels'] = str_labels
    plt.figure(figsize=(10, 8))
    # print(sorted(set(labels)))
    # for label in sorted(set(labels)):
    #     x = data[data['labels'] == label]["x"]
    #     y = data[data['labels'] == label]["y"]
    #     plt.scatter(x, y, s=5)

    sns.relplot(x="x", y="y", hue="labels", data=data, size="labels", sizes=(10, 10), style="labels")
    plt.savefig("images/cluster_ipv62vec.png", dpi=300)





if __name__ == "__main__":

    train()
    # word_tsne_picture()
    # dim_reduced_data = address_tsne_picture()


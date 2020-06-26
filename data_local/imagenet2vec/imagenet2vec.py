import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import string
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine as cos_dist
import numpy as np
import pandas as pd
import json
import tensorflow.keras as keras
from sentence_transformers import SentenceTransformer
plt.rcParams.update({'font.size': 1})


def load_imagenet_classes(fname='groupings-csv/imagenet1k_class2labels.txt'):
    with open(fname,'r') as f:
        data = []
        for l in f.readlines():
            l = l.split(": ")[1]  # remove the number and :, get only description
            l = l.replace('\n', '')  # remove the \n at the end of each entry
            l = l.replace('{', '').replace('}', '')
            # l = l.translate(str.maketrans('', '', string.punctuation))  # remove all punct
            data.append(l)
    return data


def sentenceBert(sentences):
    """
        ref: https://github.com/UKPLab/sentence-transformers
    """
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    emb = model.encode(sentences)
    np.save('data_local/imagenet2vec/imagenet2vec.npy', emb)
    # (1000, 768)
    print('saved.')


def emb_for_1k_imagenet_classes():
    # first load 1000 imagenet class labels as sentences
    # ['class1', 'class2', 'classn']
    sentences = load_imagenet_classes()
    # use sentenceBert to create embeddings
    # will result in (1000, 768) embedding matrix
    sentenceBert(sentences)


def emb_for_200_adv_classes():
    # (1000, 768)
    emb_1k = np.load('data/imagenet2vec/imagenet2vec_1k.npy')
    df = pd.read_csv('groupings-csv/imagenetA_Imagenet.csv', usecols=['wnid', 'description', 'idx'])
    sorted_indices = np.argsort([i for i in df['wnid']])
    group_indices = np.array([i for i in df['idx']])[sorted_indices]
    # (200, 768)
    emb_200 = emb_1k[group_indices, :]
    print(emb_200.shape)
    np.save('data/imagenet2vec/imagenet2vec_200.npy', emb_200)


################################################################################
# checks
def tsne_check_sentenceBert(classes='1k'):
    """
        sanity check for sentenceBert, see if results make sense on tsne plot.
    """
    emb = np.load('data/imagenet2vec/imagenet2vec_{classes}.npy'.format(classes=classes))
    emb = PCA(n_components=50).fit_transform(emb)
    best_kl_loss = None
    best_Y = None
    for i in range(5):
        model = TSNE(n_components=2, n_iter=10000)
        Y = model.fit_transform(emb)
        kl_loss = model.kl_divergence_
        print('iter = [{iter}], kl_loss = [{loss}]'.format(iter=i, loss=kl_loss))
        if best_kl_loss is None:
            best_kl_loss = kl_loss
            best_Y = Y
        else:
            if kl_loss < best_kl_loss:
                best_kl_loss = kl_loss
                best_Y = Y
    print('best_kl_loss = ', best_kl_loss)
    # plot tsne results
    all_x = best_Y[:, 0]
    all_y = best_Y[:, 1]
    fig = plt.figure(dpi=1000)
    ax = fig.add_subplot(1, 1, 1)
    default_s = plt.rcParams['lines.markersize'] ** 2
    ax.scatter(all_x, all_y, alpha=0.2, s=default_s/2)
    if classes == '1k':
        # annotate = range(1000)  # TODO: perhaps there's a better way doing this?
        data = load_imagenet_classes()
        data = [i.split(',')[0] for i in data]
        annotate = data
        print(len(annotate))
        # print(annotate)
        # exit()
    if classes == '200':
        df = pd.read_csv('groupings-csv/imagenetA_Imagenet.csv', usecols=['wnid', 'description', 'idx'])
        sorted_indices = np.argsort([i for i in df['wnid']])
        group_descriptions = np.array([i for i in df['description']])[sorted_indices]
        annotate = group_descriptions
    for i, txt in enumerate(annotate):
        text = ax.annotate(txt, (all_x[i], all_y[i]))
        text.set_alpha(0.4)
    plt.savefig('data/imagenet2vec/imagenet2vec_{classes}.pdf'.format(classes=classes))


if __name__ == '__main__':
    emb_for_1k_imagenet_classes()
    # emb_for_200_adv_classes()


    #######
    # tsne_check_sentenceBert(classes='1k')

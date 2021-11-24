import pandas as pd
import dataget
import gensim.downloader as api
import numpy as np
from scipy import io
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

mnist = load_digits()

_, _, f_mnist, f_mnist_target = dataget.image.fashion_mnist().get()
f_mnist = f_mnist.reshape(-1, 28*28)

mat = io.loadmat('COIL20.mat')
coil = mat['X']

google_news = api.load('word2vec-google-news-300').vectors
google_news = google_news[np.random.randint(google_news.shape[0], size = int(google_news.shape[0]/50)), :]

tsne = TSNE(n_jobs = -1)
pca = PCA(n_components = 2)
umap = UMAP(random_state = 2)


datasets = [coil, mnist.data, f_mnist, google_news]
datasetsName = ['COIL', 'MNIST', 'F-MNIST', 'GOOGLE NEWS']
datasetsTarget = [mat['Y'], mnist.target, f_mnist_target, None]
algs = [umap, tsne, pca]
algsName = ['UMAP', 'TSNE', 'PCA']


fig = plt.figure(figsize=(20, 20), constrained_layout=False)
gs = gridspec.GridSpec(nrows=3, ncols=4, wspace = 0.0, hspace = 0.0)

for i in range(3):
    for j in range(4):
        print(f"Alg: {algsName[i]}, dataset: {datasetsName[j]}")
        print()
        ax = fig.add_subplot(gs[i, j])
        ax.set_xlabel(datasetsName[j])
        ax.xaxis.set_label_position('top')
        ax.set_ylabel(algsName[i])
        ax.set_xticks([])
        ax.set_yticks([])
        embeddings = algs[i].fit_transform(datasets[j])
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        ax.scatter(x, y, c = datasetsTarget[j], cmap = 'Spectral', marker = '.')

plt.show()

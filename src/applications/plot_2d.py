import numpy as np
from core.util import get_scale, spectral_clustering
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

def process(x_spectralnet, y_spectralnet, data, params):
    # UNPACK DATA
    x_train, y_train, x_val, y_val, x_test, y_test = data['spectral']['train_and_test']

    # concatenate
    x = np.concatenate([x_train, x_val, x_test], axis=0)
    y = np.concatenate([y_train, y_val, y_test], axis=0)

    # PERFORM SPECTRAL CLUSTERING ON DATA

    # get eigenvalues and eigenvectors
    scale = get_scale(x, params['batch_size'], params['scale_nbr'])
    values, vectors = spectral_clustering(x, scale, params['n_nbrs'], params['affinity'])

    # sort, then store the top n_clusters=2
    values_idx = np.argsort(values)
    x_spectral_clustering = vectors[:, values_idx[:params['n_clusters']]]

    # do kmeans clustering in this subspace
    y_spectral_clustering = KMeans(n_clusters=params['n_clusters']).fit_predict(vectors[:, values_idx[:params['n_clusters']]])

    # PLOT RESULTS

    # plot spectral net clustering
    fig2 = plt.figure()
    if x.shape[1] == 2:
        ax1 = fig2.add_subplot(311)
        ax1.scatter(x[:, 0], x[:, 1],
                    alpha=0.5, s=20, cmap='rainbow', c=y_spectralnet, lw=0)
    ax1.set_title("x colored by net prediction")

    # plot spectral clustering clusters
    if x.shape[1] == 2:
        ax2 = fig2.add_subplot(313)
        ax2.scatter(x[:, 0], x[:, 1],
                    alpha=0.5, s=20, cmap='rainbow', c=y_spectral_clustering, lw=0)
    ax2.set_title("x colored by specral clustering")

    # plot histogram of eigenvectors
    fig3 = plt.figure()
    ax1 = fig3.add_subplot(212)
    ax1.hist(x_spectral_clustering)
    ax1.set_title("histogram of true eigenvectors")
    ax2 = fig3.add_subplot(211)
    ax2.hist(x_spectralnet)
    ax2.set_title("histogram of net outputs")

    # plot eigenvectors
    y_idx = np.argsort(y)
    fig4 = plt.figure()
    ax1 = fig4.add_subplot(212)
    ax1.plot(x_spectral_clustering[y_idx])
    ax1.set_title("plot of true eigenvectors")
    ax2 = fig4.add_subplot(211)
    ax2.plot(x_spectralnet[y_idx])
    ax2.set_title("plot of net outputs")

    plt.draw()
    plt.show()

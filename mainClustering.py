from OwnPCA import pca_reduce_adult, pca_reduce_vowel, pca_reduce_pen_based
from load_datasets import load_adult, load_vowel, load_pen_based
from sklearn.cluster import AgglomerativeClustering, FeatureAgglomeration
from kmeans import kmeans
import matplotlib.pyplot as plt
from metrics import run_metrics
import numpy as np
import time

# Main file to run all clustering algorithms in original and reduced data sets
# Define the file you want to apply PCA to
file = str(input(f"Write the name of file to run PCA to (vowel, pen-based, adult): "))

dim_plot = 2

# datasets are loaded either, the original or pca reduced, with corresponding parameters obtained from analysis
if file == 'vowel':
    plot_feat1 = 1
    plot_feat2 = 3
    k = 11
    dim = 2
    x_transformed = pca_reduce_vowel(dim)
    data,y = load_vowel()
if file == 'pen-based':
    plot_feat1 = 0
    plot_feat2 = 15
    k = 10
    dim = 5
    x_transformed = pca_reduce_pen_based(dim)
    data,y = load_pen_based()
if file == 'adult':
    plot_feat1 = 12
    plot_feat2 = 13
    k = 2
    dim = 6
    x_transformed = pca_reduce_adult(dim)
    x_transformed = x_transformed[:int(len(x_transformed)*0.6)]
    data,y = load_adult()
    data,y = data[:int(len(data)*0.6)], y[:int(len(data)*0.6)]


# plotting of clusters
def plot_clusters(np_data, class_labels, dim, algorithm):
    """
    Function to plot the clusters obtained
    """
    if 'original' in algorithm.lower():
        if dim == 2:
            plt.scatter(np_data[:, plot_feat1], np_data[:, plot_feat2], c=class_labels)
            plt.title(algorithm)
            plt.xlabel(f"Feature {plot_feat1}")
            plt.ylabel(f"Feature {plot_feat2}")
            plt.show()

    else:
        if dim == 2:
            plt.scatter(np_data[:, 0], np_data[:, 1], c=class_labels)
            plt.title(algorithm)
            plt.xlabel("PC 1")
            plt.ylabel("PC 2")
            plt.show()

        if dim == 3:
            fig = plt.figure()
            ax = plt.axes(projection='3d')

            ax.scatter(np_data[:, 0], np_data[:, 1], np_data[:, 2], c=class_labels)

          # for k = 3

            plt.title(algorithm)
            plt.xlabel("PC 1")
            plt.ylabel("PC 2")
            plt.clabel("PC 3")
            plt.show()


np_data = np.array(data)
np_data_transformed = np.array(x_transformed)

# kmeans
# original
init_time = time.time()
centroids, kmeans_label = kmeans(k, np_data, 'means')
end_time = time.time()

# metrics
ars, hs, accsc, fs = run_metrics(y, kmeans_label)
print(f"original K-means:\n"
      f"runtime: {round(end_time - init_time,2)}s\n"
      f"Accuracy score: {str(accsc)}\n"
      f"F-score: {str(fs)}\n")
# plot results
plot_clusters(np_data, kmeans_label, dim_plot, 'K-Means on original Data')

# transformed data PCA
init_time = time.time()
centroids, kmeans_label_pca = kmeans(k, np_data_transformed, 'means')
end_time = time.time()
# metrics
ars, hs, accsc, fs = run_metrics(y, kmeans_label_pca)
print("K-means PCA: \n"
      f"runtime: {round(end_time - init_time,2)}s\n"
      f"Accuracy score: {str(accsc)}\n"
      f"F-score: {str(fs)}\n")
# plot results
plot_clusters(np_data_transformed, kmeans_label_pca, dim_plot, 'K-Means on PCA reduced Data')

# Agglomerative clustering
# original
init_time = time.time()
agg_clustering = AgglomerativeClustering(n_clusters=k).fit(np_data)
end_time = time.time()

agg_clust_labels = agg_clustering.labels_
# metrics
ars, hs, accsc, fs = run_metrics(y, agg_clust_labels)
print("Agglomerative Clustering original: \n"
      f"runtime: {round(end_time - init_time,2)}s\n"
      f"Accuracy score: {str(accsc)}\n"
      f"F-score: {str(fs)}\n")
# plot results
plot_clusters(np_data, agg_clust_labels, dim_plot, 'Agglomerative Clustering on original Data')

# transformed data PCA
init_time = time.time()
agg_clustering_pca = AgglomerativeClustering(n_clusters=k).fit(np_data_transformed)
end_time = time.time()
agg_clust_labels_pca = agg_clustering_pca.labels_
# metrics
ars, hs, accsc, fs = run_metrics(y, agg_clust_labels_pca)
print("Agglomerative Clustering PCA: \n"
      f"runtime: {round(end_time - init_time,2)}s\n"
      f"Accuracy score: {str(accsc)}\n"
      f"F-score: {str(fs)}\n")
# plot results
plot_clusters(np_data_transformed, agg_clust_labels_pca, dim_plot, 'Agglomerative Clustering on PCA reduced Data')

# FeatureAgglomeration
np_data_fa = np.array(data)
feat_agg_clustering = FeatureAgglomeration(dim)
feat_agg_clustering.fit(np_data_fa)
X_fa_reduced = feat_agg_clustering.transform(np_data_fa)

# transformed data with Feature Agglomeration
init_time = time.time()
centroids, kmeans_label_fa = kmeans(k, X_fa_reduced, 'means')
end_time = time.time()

# metrics
ars, hs, accsc, fs = run_metrics(y, kmeans_label_fa)

print("Feature Agglomeration K-means: \n"
      f"runtime: {round(end_time - init_time,2)}s\n"
      f"Accuracy score: {str(accsc)}\n"
      f"F-score: {str(fs)}\n")
# plot results
plot_clusters(X_fa_reduced, kmeans_label_fa, dim_plot, 'K-Means on Feature Agglomeration reduced Data')

# data with Feature Agglomeration
init_time = time.time()
agg_clustering = AgglomerativeClustering(n_clusters=k).fit(X_fa_reduced)
end_time = time.time()
agg_clust_labels_fa = agg_clustering.labels_
# metrics
ars, hs, accsc, fs = run_metrics(y, agg_clust_labels_fa)
print("Agglomerative Clustering Feature Agglomeration: \n"
      f"runtime: {round(end_time - init_time,2)}s\n"
      f"Accuracy score: {str(accsc)}\n"
      f"F-score: {str(fs)}\n")
# plot results
plot_clusters(X_fa_reduced, agg_clust_labels_fa, dim_plot,
              'Agglomerative clustering on Feature Agglomeration reduced Data')

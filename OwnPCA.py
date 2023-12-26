# Principal Component Analysis algorithm
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from load_datasets import load_adult, load_vowel, load_pen_based


def find_best_k(eigenvalues):
    """
    Create a plot of the cumulative percentage of explained
    variance for each principal component.
    """
    plt.plot(range(1, len(eigenvalues)+1), np.cumsum(100 * eigenvalues / np.sum(eigenvalues)))
    plt.scatter(range(1, len(eigenvalues)+1), np.cumsum(100 * eigenvalues / np.sum(eigenvalues)))
    plt.axhline(y=80, color='r', linestyle='-', lw=0.5)
    plt.title('Find best k')
    plt.xlabel("Principal Components")
    plt.ylabel("Cumulative % of explained variance")
    plt.show()


def plot_transX(transX, dim, classes=None):
    """
    Plot the dimension reduced data (transX)
    Depending on the number of dimensions (dim) different plots will
    be created (2- or 3-D).
    If available, the classes can define the color of each datapoint.
    """
    # Plot the dimension reduced data
    # If the number of dimensions = 3 create a 3D plot
    if dim == 3:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(transX[:, 0], transX[:, 1], transX[:, 2], c=classes)  # for k = 3
        plt.title("PCA Result")
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.clabel("PC 3")
        plt.show()
    elif dim == 1:
        plt.scatter(transX[:, 0], np.zeros(len(transX[:, 0])), c=classes)  # for k = 1
        plt.title("PCA Result")
        plt.xlabel("PC 1")
        plt.show()
    # For the rest of the cases (k ≠ 3 and k ≠ 1) create a 2D plot with the 2 most relevant principal components
    else:
        plt.scatter(transX[:, 0], transX[:, 1], c=classes)  # for k = 2
        plt.title("PCA Result")
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.show()


def plot_original_adult(Xadult, classes=None):
    """
    Plot the original adult data set without the dim reduction.
    If available, the classes can define the color of each datapoint.
    """
    plt.scatter(Xadult[:, 12], Xadult[:, 13], c=classes)
    plt.title("Original data - Adult dataset")
    plt.xlabel("Feature 12")
    plt.ylabel("Feature 13")
    plt.show()


def plot_original_vowel(Xvowel, classes=None):
    """
    Plot the original vowel data set without the dim reduction.
    If available, the classes can define the color of each datapoint.
    """
    plt.scatter(Xvowel[:, 1], Xvowel[:, 3], c=classes)
    plt.title("Original data - Vowel dataset")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 3")
    plt.show()


def plot_original_pen_based(Xpb, classes=None):
    """
    Plot the original pne-based data setwithout the dim reduction.
    If available, the classes can define the color of each datapoint.
    """
    plt.scatter(Xpb[:, 0], Xpb[:, 15], c=classes)
    plt.title("Original data - Pen-based dataset")
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 15")
    plt.show()


def pca_adult(k=2):
    """
    Run our own PCA code for the adult dataset and show relevant results
    """
    df, classes = load_adult()  # Load adult data into a df
    Xraw = df.values  # Get only values from df

    # Plot the original dataset (without dim reduction)
    plot_original_adult(Xraw)

    mean_vec = np.mean(Xraw, axis=0)  # mean vector
    X = Xraw - mean_vec  # Subtract the mean of each feature

    cov_matrix = np.cov(X.T)  # Compute the covariance matrix
    print("Covariance Matrix: ")
    print(cov_matrix)  # print the cov matrix

    eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)  # Calculate eigenvalues and eigenvectors from cov matrix
    print("\nThe eigen values are: ")
    print(eigen_values)  # print the eigen values
    print("\nThe eigen vectors are: ")
    print(eigen_vectors)  # print the eigen vectors

    # Sort the eigenvectors in eigenvalues descending order and show them
    sort_ind = np.flip(np.argsort(eigen_values))  # get args from sorted eigenvalues in desc order (from max to min)
    sorted_eigval = eigen_values[sort_ind]  # sort eigenvalues with given args
    feature_vector = eigen_vectors[:, sort_ind]  # sort eigenvectors with given args

    # Create a plot to find best k value
    find_best_k(sorted_eigval)

    reduced_feature_vector = feature_vector[:, 0:k]  # k first eigenvectors
    reduced_feature_values = sorted_eigval[0:k]  # k first eigenvalues

    print("\nThe first k eigenvectors are: ")
    print(reduced_feature_vector)  # print the reduced feature vector
    print("\nAnd their corresponding eigenvalues are: ")
    print(reduced_feature_values)  # print the reduced feature vector eigenvalues

    # Transform data
    transformed_X = np.dot(X, reduced_feature_vector)  # X(n x p); reduced_feature_vector(p x k); p=features, k=dim

    # Reconstruct data
    X_recons = np.dot(transformed_X, reduced_feature_vector.T)  # transformed_X(k x n); reduced_feature_vector(p x k)
    X_original = X_recons + mean_vec

    # Plot the dimension reduced data
    plot_transX(transformed_X, k)

    plt.scatter(X_original[:, 1], X_original[:, 9])  # Plot the reconstructed original dataset
    plt.title("Reconstructed original data - vowel dataset")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 9")
    plt.show()


def pca_vowel(k=2):
    """
    Run our own PCA code for the vowel dataset and show relevant results
    """
    df, classes = load_vowel()  # Load adult data into a df
    Xraw = df.values  # Get only values from df

    # Plot the original dataset (without dim reduction)
    plot_original_vowel(Xraw)

    mean_vec = np.mean(Xraw, axis=0)  # mean vector
    X = Xraw - mean_vec  # Subtract the mean of each feature

    cov_matrix = np.cov(X.T)  # Compute the covariance matrix
    print("Covariance Matrix: ")
    print(cov_matrix)  # print the cov matrix

    eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)  # Calculate eigenvalues and eigenvectors from cov matrix
    print("\nThe eigen values are: ")
    print(eigen_values)  # print the eigen values
    print("\nThe eigen vectors are: ")
    print(eigen_vectors)  # print the eigen vectors

    # Sort the eigenvectors in eigenvalues descending order and show them
    sort_ind = np.flip(np.argsort(eigen_values))  # get args from sorted eigenvalues in desc order (from max to min)
    sorted_eigval = eigen_values[sort_ind]  # sort eigenvalues with given args
    feature_vector = eigen_vectors[:, sort_ind]  # sort eigenvectors with given args

    # Create a plot to find best k value
    find_best_k(sorted_eigval)

    reduced_feature_vector = feature_vector[:, 0:k]  # k first eigenvectors
    reduced_feature_values = sorted_eigval[0:k]  # k first eigenvalues

    print("\nThe first k eigenvectors are: ")
    print(reduced_feature_vector)  # print the reduced feature vector
    print("\nAnd their corresponding eigenvalues are: ")
    print(reduced_feature_values)  # print the reduced feature vector eigenvalues

    # Transform data
    transformed_X = np.dot(X, reduced_feature_vector)  # X(n x p); reduced_feature_vector(p x k); p=features, k=dim

    # Reconstruct data
    X_recons = np.dot(transformed_X, reduced_feature_vector.T)  # transformed_X(k x n); reduced_feature_vector(p x k)
    X_original = X_recons + mean_vec

    # Plot the dimension reduced data
    plot_transX(transformed_X, k)

    plt.scatter(X_original[:, 1], X_original[:, 3])  # Plot the reconstructed original dataset
    plt.title("Reconstructed original data - vowel dataset")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 3")
    plt.show()


def pca_pen_based(k=2):
    """
    Run our own PCA code for the pen-based dataset and show relevant results
    """
    df, classes = load_pen_based()  # Load pen-based data into a df
    Xraw = df.values  # Get only values from df

    # Plot the original dataset (without dim reduction)
    plot_original_pen_based(Xraw)

    mean_vec = np.mean(Xraw, axis=0)  # mean vector
    X = Xraw - mean_vec  # Subtract the mean of each feature

    cov_matrix = np.cov(X.T)  # Compute the covariance matrix
    print("Covariance Matrix: ")
    print(cov_matrix)  # print the cov matrix

    eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)  # Calculate eigenvalues and eigenvectors from cov matrix
    print("\nThe eigen values are: ")
    print(eigen_values)  # print the eigen values
    print("\nThe eigen vectors are: ")
    print(eigen_vectors)  # print the eigen vectors

    # Sort the eigenvectors in eigenvalues descending order and show them
    sort_ind = np.flip(np.argsort(eigen_values))  # get args from sorted eigenvalues in desc order (from max to min)
    sorted_eigval = eigen_values[sort_ind]  # sort eigenvalues with given args
    feature_vector = eigen_vectors[:, sort_ind]  # sort eigenvectors with given args

    # Create a plot to find best k value
    find_best_k(sorted_eigval)

    reduced_feature_vector = feature_vector[:, 0:k]  # k first eigenvectors
    reduced_feature_values = sorted_eigval[0:k]  # k first eigenvalues

    print("\nThe first k eigenvectors are: ")
    print(reduced_feature_vector)  # print the reduced feature vector
    print("\nAnd their corresponding eigenvalues are: ")
    print(reduced_feature_values)  # print the reduced feature vector eigenvalues

    # Transform data
    transformed_X = np.dot(X, reduced_feature_vector)  # X(n x p); reduced_feature_vector(p x k); p=features, k=dim

    # Reconstruct data
    X_recons = np.dot(transformed_X, reduced_feature_vector.T)  # transformed_X(k x n); reduced_feature_vector(p x k)
    X_original = X_recons + mean_vec

    # Plot the dimension reduced data
    plot_transX(transformed_X, k)

    plt.scatter(X_original[:, 0], X_original[:, 15])  # Plot the reconstructed original dataset
    plt.title("Reconstructed original data - pen-based dataset")
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 15")
    plt.show()


def pca_reduce_adult(k=2):
    """
    Run our own PCA code for the adult dataset
    """
    df, classes = load_adult()  # Load adult data into a df
    Xraw = df.values  # Get only values from df

    mean_vec = np.mean(Xraw, axis=0)  # mean vector
    X = Xraw - mean_vec  # Subtract the mean of each feature

    cov_matrix = np.cov(X.T)  # Compute the covariance matrix
    eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)  # Calculate eigenvalues and eigenvectors from cov matrix

    # Sort the eigenvectors in eigenvalues descending order and show them
    sort_ind = np.flip(np.argsort(eigen_values))  # get args from sorted eigenvalues in desc order (from max to min)
    sorted_eigval = eigen_values[sort_ind]  # sort eigenvalues with given args
    feature_vector = eigen_vectors[:, sort_ind]  # sort eigenvectors with given args

    reduced_feature_vector = feature_vector[:, 0:k]  # k first eigenvectors
    reduced_feature_values = sorted_eigval[0:k]  # k first eigenvalues

    # Transform data
    transformed_X = np.dot(X, reduced_feature_vector)  # X(n x p); reduced_feature_vector(p x k); p=features, k=dim

    # Reconstruct data
    X_recons = np.dot(transformed_X, reduced_feature_vector.T)  # transformed_X(k x n); reduced_feature_vector(p x k)
    X_original = X_recons + mean_vec

    return transformed_X


def pca_reduce_vowel(k=2):
    """
    Run our own PCA code for the vowel dataset
    """
    df, classes = load_vowel()  # Load adult data into a df
    Xraw = df.values  # Get only values from df

    mean_vec = np.mean(Xraw, axis=0)  # mean vector
    X = Xraw - mean_vec  # Subtract the mean of each feature

    cov_matrix = np.cov(X.T)  # Compute the covariance matrix
    eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)  # Calculate eigenvalues and eigenvectors from cov matrix

    # Sort the eigenvectors in eigenvalues descending order and show them
    sort_ind = np.flip(np.argsort(eigen_values))  # get args from sorted eigenvalues in desc order (from max to min)
    sorted_eigval = eigen_values[sort_ind]  # sort eigenvalues with given args
    feature_vector = eigen_vectors[:, sort_ind]  # sort eigenvectors with given args

    reduced_feature_vector = feature_vector[:, 0:k]  # k first eigenvectors
    reduced_feature_values = sorted_eigval[0:k]  # k first eigenvalues

    # Transform data
    transformed_X = np.dot(X, reduced_feature_vector)  # X(n x p); reduced_feature_vector(p x k); p=features, k=dim

    # Reconstruct data
    X_recons = np.dot(transformed_X, reduced_feature_vector.T)  # transformed_X(k x n); reduced_feature_vector(p x k)
    X_original = X_recons + mean_vec

    return transformed_X


def pca_reduce_pen_based(k=2):
    """
    Run our own PCA code for the pen-based dataset
    """
    df, classes = load_pen_based()  # Load pen-based data into a df
    Xraw = df.values  # Get only values from df

    mean_vec = np.mean(Xraw, axis=0)  # mean vector
    X = Xraw - mean_vec  # Subtract the mean of each feature

    cov_matrix = np.cov(X.T)  # Compute the covariance matrix
    eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)  # Calculate eigenvalues and eigenvectors from cov matrix

    # Sort the eigenvectors in eigenvalues descending order and show them
    sort_ind = np.flip(np.argsort(eigen_values))  # get args from sorted eigenvalues in desc order (from max to min)
    sorted_eigval = eigen_values[sort_ind]  # sort eigenvalues with given args
    feature_vector = eigen_vectors[:, sort_ind]  # sort eigenvectors with given args

    reduced_feature_vector = feature_vector[:, 0:k]  # k first eigenvectors
    reduced_feature_values = sorted_eigval[0:k]  # k first eigenvalues

    # Transform data
    transformed_X = np.dot(X, reduced_feature_vector)  # X(n x p); reduced_feature_vector(p x k); p=features, k=dim

    # Reconstruct data
    X_recons = np.dot(transformed_X, reduced_feature_vector.T)  # transformed_X(k x n); reduced_feature_vector(p x k)
    X_original = X_recons + mean_vec

    return transformed_X


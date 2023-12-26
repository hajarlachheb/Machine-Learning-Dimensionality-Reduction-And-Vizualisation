# Principal Component Analysis algorithm
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from load_datasets import load_adult, load_vowel, load_pen_based


def pca_adult(k=2):
    df, classes = load_adult()  # Load adult data into a df
    Xraw = df.values  # Get only values from df

    plt.scatter(Xraw[:, 1], Xraw[:, 9], c=classes)  # Plot the original dataset (without dim reduction)
    plt.title("Original data - adult dataset")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 9")
    plt.show()

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
    if k == 2:
        plt.scatter(transformed_X[:, 0], transformed_X[:, 1], c=classes)  # for k = 2
        plt.title("PCA - adult dataset")
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.show()
    elif k == 3:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(transformed_X[:, 0], transformed_X[:, 1], transformed_X[:, 2], c=classes)  # for k = 3
        plt.title("PCA - adult dataset")
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.clabel("PC 3")
        plt.show()
    else:
        print("\nPCA dim reduced data cannot be plotted for k ≠ {2, 3}")

    plt.scatter(X_original[:, 1], X_original[:, 9], c=classes)  # Plot the reconstructed original dataset
    plt.title("Reconstructed original data - vowel dataset")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 9")
    plt.show()
    return transformed_X

def pca_vowel(k=2):
    df, classes = load_vowel()  # Load adult data into a df
    Xraw = df.values  # Get only values from df

    plt.scatter(Xraw[:, 1], Xraw[:, 3], c=classes)  # Plot the original dataset (without dim reduction)
    plt.title("Original data - vowel dataset")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 3")
    plt.show()

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

    reduced_feature_vector = feature_vector[:, 0:k]  # k first eigenvectors
    reduced_feature_values = sorted_eigval[0:k]  # k first eigenvalues

    print("\nThe first k eigenvectors are: ")
    print(reduced_feature_vector[:,0])  # print the reduced feature vector
    print(reduced_feature_vector[:,1])
    print(reduced_feature_vector[:,2])
    print("\nAnd their corresponding eigenvalues are: ")
    print(reduced_feature_values)  # print the reduced feature vector eigenvalues

    # Transform data
    transformed_X = np.dot(X, reduced_feature_vector)  # X(n x p); reduced_feature_vector(p x k); p=features, k=dim

    # Reconstruct data
    X_recons = np.dot(transformed_X, reduced_feature_vector.T)  # transformed_X(k x n); reduced_feature_vector(p x k)
    X_original = X_recons + mean_vec

    # Plot the dimension reduced data
    if k == 2:
        plt.scatter(transformed_X[:, 0], transformed_X[:, 1], c=classes)  # for k = 2
        plt.title("PCA - vowel dataset")
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.show()
    elif k == 3:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(transformed_X[:, 0], transformed_X[:, 1], transformed_X[:, 2], c=classes)  # for k = 3
        plt.title("PCA - vowel dataset")
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.clabel("PC 3")
        plt.show()
    else:
        print("\nPCA dim reduced data cannot be plotted for k ≠ {2, 3}")

    plt.scatter(X_original[:, 1], X_original[:, 3], c=classes)  # Plot the reconstructed original dataset
    plt.title("Reconstructed original data - vowel dataset")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 3")
    plt.show()
    return transformed_X


def pca_pen_based(k=2):
    df, classes = load_pen_based()  # Load adult data into a df
    Xraw = df.values  # Get only values from df

    plt.scatter(Xraw[:, 0], Xraw[:, 15], c=classes)  # Plot the original dataset (without dim reduction)
    plt.title("Original data - pen-based dataset")
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 15")
    plt.show()

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
    if k == 2:
        plt.scatter(transformed_X[:, 0], transformed_X[:, 1], c=classes)  # for k = 2
        plt.title("PCA - pen-based dataset")
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.show()
    elif k == 3:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(transformed_X[:, 0], transformed_X[:, 1], transformed_X[:, 2], c=classes)  # for k = 3
        plt.title("PCA - pen-based dataset")
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.clabel("PC 3")
        plt.show()
    else:
        print("\nPCA dim reduced data cannot be plotted for k ≠ {2, 3}")

    plt.scatter(X_original[:, 0], X_original[:, 15], c=classes)  # Plot the reconstructed original dataset
    plt.title("Reconstructed original data - pen-based dataset")
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 15")
    plt.show()

    return transformed_X




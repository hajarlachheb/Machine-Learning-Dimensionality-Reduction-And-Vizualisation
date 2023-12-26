import OwnPCA
from OwnPCA import pca_adult, pca_vowel, pca_pen_based
from sklearn.decomposition import PCA, IncrementalPCA
from load_datasets import load_adult, load_vowel, load_pen_based
import numpy as np
import matplotlib.pyplot as plt

# Choose the PCA to perform:
option = int(input("Choose the number of the PCA option you'd like to run:"
                   "\n 1: Our own PCA"
                   "\n 2: Sklearn's PCA"
                   "\n 3: Sklearn's Incremental PCA"
                   "\nWrite the desired option (number): "))

# Define the file you want to apply PCA to
file = str(input(f"Write the name of file to run PCA to (vowel, pen-based, adult): "))

# Define the number of components (k)
k = int(input(f"Write the desired number of components to perform the dimensionality reduction k: "))


# Run our own PCA code
def run_own_PCA(f, k):
    """
    Function to run our PCA code for a desired data set and number of dimensions and show
    relevant information
    :param f: file in which to perform the PCA
    :param k: dimensions after the dimensionality reduction
    """
    if f == 'pen-based':
        if k < 1 or k > 15:
            raise ValueError(
                'To perform dimensionality reduction k must be in range [1,15] for pen-based and k was {}'.format(k))
        pca_pen_based(k)
    elif f == 'vowel':
        if k < 1 or k > 12:
            raise ValueError(
                'To perform dimensionality reduction k must be in range [1,12] for vowel and k was {}'.format(k))
        pca_vowel(k)
    elif f == 'adult':
        if k < 1 or k > 13:
            raise ValueError(
                'To perform dimensionality reduction k must be in range [1,13] for vowel and k was {}'.format(k))
        pca_adult(k)
    else:
        raise ValueError('Unknown dataset {}'.format(file))


# Run sklearn PCA
def run_sklearn_PCA(f, k):
    """
        Function to run sklearn's PCA code for a desired data set and number of dimensions and show
        relevant information
        :param f: file in which to perform the PCA
        :param k: dimensions after the dimensionality reduction
        """
    if f == 'pen-based':
        df, cl = load_pen_based()
    elif f == 'vowel':
        df, cl = load_vowel()
    elif f == 'adult':
        df, cl = load_adult()
    else:
        raise ValueError('Unknown dataset {}'.format(file))

    pca = PCA(n_components=k)
    transX = pca.fit_transform(df.values, cl)
    print("\nPCA sklearn: ")
    print("Eigenvalues: ")
    print(pca.explained_variance_)
    print("Eigenvectors: ")
    print(pca.components_)

    # Plot the dimension reduced data
    # If the number of dimensions = 3 create a 3D plot
    if k == 3:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(transX[:, 0], transX[:, 1], transX[:, 2])  # for k = 3
        plt.title("PCA Result")
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.clabel("PC 3")
        plt.show()
    elif k == 1:
        plt.scatter(transX[:, 0], np.zeros(len(transX[:, 0])))  # for k = 1
        plt.title("sklearn's PCA Result")
        plt.xlabel("PC 1")
        plt.show()
    # For the rest of the cases (k ≠ 3 and k ≠ 1) create a 2D plot with the 2 most relevant principal components
    else:
        plt.scatter(transX[:, 0], transX[:, 1])  # for k = 2
        plt.title("sklear's PCA Result")
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.show()


# Run sklearn Incremental PCA
def run_sklearn_incPCA(f, k):
    """
        Function to run sklearn's Incremental PCA code for a desired data set and number of dimensions and show
        relevant information
        :param f: file in which to perform the PCA
        :param k: dimensions after the dimensionality reduction
        """
    if f == 'pen-based':
        df, cl = load_pen_based()
    elif f == 'vowel':
        df, cl = load_vowel()
    elif f == 'adult':
        df, cl = load_adult()
    else:
        raise ValueError('Unknown dataset {}'.format(file))

    inc_pca = IncrementalPCA(n_components=k)
    inc_pca.fit(df.values)
    transX = inc_pca.transform(df.values)
    print("\nIncremental PCA sklearn: ")
    print("Eigenvectors: ")
    print(inc_pca.components_)
    print("Eigenvalues: ")
    print(inc_pca.explained_variance_)

    # Plot the dimension reduced data
    # If the number of dimensions = 3 create a 3D plot
    if k == 3:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(transX[:, 0], transX[:, 1], transX[:, 2])  # for k = 3
        plt.title("PCA Result")
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.clabel("PC 3")
        plt.show()
    elif k == 1:
        plt.scatter(transX[:, 0], np.zeros(len(transX[:, 0])))  # for k = 1
        plt.title("sklearn's Incremental PCA Result")
        plt.xlabel("PC 1")
        plt.show()
    # For the rest of the cases (k ≠ 3 and k ≠ 1) create a 2D plot with the 2 most relevant principal components
    else:
        plt.scatter(transX[:, 0], transX[:, 1])  # for k = 2
        plt.title("sklear's Incremental PCA Result")
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.show()


if option == 1:
    # Run own PCA
    run_own_PCA(file, k)

elif option == 2:
    # Run sklearn's PCA
    run_sklearn_PCA(file, k)

elif option == 3:
    # Run sklearn's Incremental PCA
    run_sklearn_incPCA(file, k)

else:
    raise ValueError("There is no option {}".format(option))

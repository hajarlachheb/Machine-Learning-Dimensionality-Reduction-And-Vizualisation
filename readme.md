### Team members:

## Joël Haupt
## Hajar Lachheb
## Saurabh Nagarkar
## Miquel Sugrañes Pàmies

### Running script for the first time

# Open project’s root folder in terminal:
 ```bash
cd <root_folder_of_project>/ 
```

# Create virtual environment: 
 ```bash
python3 -m venv venv/ 
```

# Open virtual environment:
 ```bash
source venv/bin/activate 
```

# Install required dependencies: 
```bash
pip install -r requirements.txt 
```

# Close virtual environment: 
```bash
deactivate
```

## Execute scripts
# Open folder in terminal: 
```bash
cd <root_folder_of_project>/
 ```

# Open the virtual environment:
```bash
source venv/bin/activate
```

# Run either our own PCA, sklearn's PCA or sklearn's Incremental PCA and show results
In this main file the different necessary options to determine which algorithm to run and the parameters required
are set by the user by writing the desired ones through console window. The different results are shown in console 
(eigenvalues, eigenvectors, covariance matrix) and via plot figures (original, transformed and reconstructed data sets).
If the number of dimensions == 3, a 3D plot is created, else a 2D plot is shown.
```bash
python3 mainPCA.py
```

# Run clustering methods (k-Means and AgglomerativeClustering) with original and reduced data (PCA and FA)
In this main file the different clustering algorithms are performed, both k-Means and AgglomerativeClustering with
the original data sets and the reduced ones resulting from the PCA and FA algorithms.
As in the previous main file, the data set with which to run the algorithms is aked to the user by console.
```bash
python3 mainClustering.py
```

# Run the visualization file which plots with PCA and t-SNE the original and reduced data sets in a 3-dimensional space
Running the following script will plot the original and the reduced data sets after making use of the t-SNE and PCA
algorithms. The plots will ben shown in a 3-dimensional space.
```bash
python3 mainVisualization.py
```
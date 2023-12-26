import load_datasets
from sklearn.decomposition import PCA
import scprep
import sklearn.manifold
import os
import pandas as pd
import numpy as np
from scipy.io.arff import loadarff
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

#-------------------------------------------------Preprocessing Adult------------------------------------#

dataset = os.path.join('datasets', 'pen-based.arff')
data = loadarff(dataset)
df_data = pd.DataFrame(data[0])

df_data.rename(columns={'capital-gain': 'capital gain', 'capital-loss': 'capital loss', 'native-country': 'country',
                            'hours-per-week': 'hours per week', 'marital-status': 'marital'}, inplace=True)
df_data['country'] = df_data['country'].replace('?', np.nan)
df_data['workclass'] = df_data['workclass'].replace('?', np.nan)
df_data['occupation'] = df_data['occupation'].replace('?', np.nan)

df_data.dropna(how='any', inplace=True)

label_encoder = LabelEncoder()

catcolumns = list(df_data.select_dtypes(include=['object']).columns)
for i in catcolumns:
    df_data[i] = label_encoder.fit_transform(df_data[i])

min_max_scaler = MinMaxScaler()

scaled_df = pd.DataFrame()

columnval = df_data.columns.values
columnval = columnval[:-1]

scaled_values = min_max_scaler.fit_transform(df_data[columnval])

for i in range(len(columnval)):
    scaled_df[columnval[i]] = scaled_values[:, i]

classes = df_data['class']

print('This is the Adult Dataset Analysis')
#-------------------------------------------ADULT DATASET--------------------------------------#
#Trying to reduce the dimensionality with PCA, it will be used later when trying t-SNE with PCA
data_pca_tsne = scprep.reduce.pca(scaled_df, n_components=3, method='dense')
#--------------------------------------------------t-SNE-----------------------------------------#
tsne_op = sklearn.manifold.TSNE(n_components=3, perplexity=50)
data_tsne = tsne_op.fit_transform(scaled_df)
#------------------------------------------------t-SNE presentation-------------------------------#
scprep.plot.scatter3d(data_tsne, classes,
                      figsize=(8,4), legend_anchor=(1,1),
                      ticks=False, label_prefix='t-SNE')
#--------------------------------------------------PCA-----------------------------------------#
pca = PCA(n_components=3)
pca_result = pca.fit_transform(scaled_df)
#------------------------------------------------PCA presentation-------------------------------#
scprep.plot.scatter3d(pca_result, classes,
                      figsize=(8,4), legend_anchor=(1,1),
                      ticks=False, label_prefix='PCA')
#------------------------------------------------t-SNE with PCA Reduction-------------------------------#
tsne_op = sklearn.manifold.TSNE(n_components=3, perplexity=50)
data_tsne = tsne_op.fit_transform(data_pca_tsne)
#------------------------------------------------t-SNE with PCA Reduction presentation-------------------------------#
scprep.plot.scatter3d(data_tsne, classes,
                      figsize=(8,4), legend_anchor=(1,1),
                      ticks=False, label_prefix='t-SNE')

#-------------------------------------------------Preprocessing Vowel------------------------------------#
# Load input dataset
dataset = os.path.join('datasets', 'vowel.arff')
data = loadarff(dataset)
df_data = pd.DataFrame(data[0])
# Preprocessing - encode categorical variables with label encoder
cols = ['Sex', 'Train_or_Test', 'Speaker_Number', 'Class']
le = LabelEncoder()
df_data[cols] = df_data[cols].apply(le.fit_transform)
# Save classes of instances
classes2 = df_data['Class']
# Drop class column from dataframe
df_data_vowel = df_data.drop(columns='Class')

print('This is the Vowel Dataset Analysis')
#-------------------------------------------VOWEL DATASET--------------------------------------#
#Trying to reduce the dimensionality with PCA, it will be used later when trying t-SNE with PCA
data_pca_tsne = scprep.reduce.pca(df_data_vowel, n_components=3, method='dense')
#--------------------------------------------------t-SNE-----------------------------------------#
tsne_op = sklearn.manifold.TSNE(n_components=3, perplexity=50)
data_tsne = tsne_op.fit_transform(df_data_vowel)
#------------------------------------------------t-SNE presentation-------------------------------#
scprep.plot.scatter3d(data_tsne, classes2,
                      figsize=(8,4), legend_anchor=(1,1),
                      ticks=False, label_prefix='t-SNE')
#--------------------------------------------------PCA-----------------------------------------#
pca = PCA(n_components=3)
pca_result = pca.fit_transform(df_data_vowel)
#------------------------------------------------PCA presentation-------------------------------#
scprep.plot.scatter3d(pca_result, classes2,
                      figsize=(8,4), legend_anchor=(1,1),
                      ticks=False, label_prefix='PCA')
#------------------------------------------------t-SNE with PCA Reduction-------------------------------#
tsne_op = sklearn.manifold.TSNE(n_components=3, perplexity=50)
data_tsne = tsne_op.fit_transform(data_pca_tsne)
#------------------------------------------------t-SNE with PCA Reduction presentation-------------------------------#
scprep.plot.scatter3d(data_tsne, classes2,
                      figsize=(8,4), legend_anchor=(1,1),
                      ticks=False, label_prefix='t-SNE')

#-------------------------------------------------Preprocessing Pen Based------------------------------------#
# Load input dataset
dataset = os.path.join('datasets', 'adult.arff')
data = loadarff(dataset)
df_data = pd.DataFrame(data[0])
# Save classes of instances
classes3 = df_data['a17'].astype(int)
# Drop class column from dataframe
df_data_pen_based = df_data.drop(columns='a17')

print('This is the Pen Based Dataset Analysis')
#-------------------------------------------PEN BASED DATASET--------------------------------------#
#Trying to reduce the dimensionality with PCA, it will be used later when trying t-SNE with PCA
data_pca_tsne = scprep.reduce.pca(df_data_pen_based, n_components=3, method='dense')
#--------------------------------------------------t-SNE-----------------------------------------#
tsne_op = sklearn.manifold.TSNE(n_components=3, perplexity=50)
data_tsne = tsne_op.fit_transform(df_data_pen_based)
#------------------------------------------------t-SNE presentation-------------------------------#
scprep.plot.scatter3d(data_tsne, classes3,
                      figsize=(8,4), legend_anchor=(1,1),
                      ticks=False, label_prefix='t-SNE')
#--------------------------------------------------PCA-----------------------------------------#
pca = PCA(n_components=3)
pca_result = pca.fit_transform(df_data_pen_based)
#------------------------------------------------PCA presentation-------------------------------#
scprep.plot.scatter3d(pca_result, classes3,
                      figsize=(8,4), legend_anchor=(1,1),
                      ticks=False, label_prefix='PCA')
#------------------------------------------------t-SNE with PCA Reduction-------------------------------#
tsne_op = sklearn.manifold.TSNE(n_components=3, perplexity=50)
data_tsne = tsne_op.fit_transform(data_pca_tsne)
#------------------------------------------------t-SNE with PCA Reduction presentation-------------------------------#
scprep.plot.scatter3d(data_tsne, classes3,
                      figsize=(8,4), legend_anchor=(1,1),
                      ticks=False, label_prefix='t-SNE')

print('We are going to plot now the original datasets')

#--------------------------PLOTTING THE ORIGINAL DATASETS-------------------------------------------#

#Adult Dataset
scprep.plot.scatter3d(scaled_df, classes,
                      figsize=(8,4), legend_anchor=(1,1),
                      ticks=False)


#Vowel Dataset
scprep.plot.scatter3d(df_data_vowel, classes2,
                      figsize=(8,4), legend_anchor=(1,1),
                      ticks=False)



#Pen Based Dataset
scprep.plot.scatter3d(df_data_pen_based, classes3,
                      figsize=(8,4), legend_anchor=(1,1),
                      ticks=False)

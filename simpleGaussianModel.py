import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA


# Load the IRIS dataset
iris = sns.load_dataset('iris')
print(iris.head()) 

# initialize seaborn
sns.set()
sns.pairplot(iris, hue='species', size=1.5, plot_kws={"s": 6}, palette='husl', markers=["o", "s", "D"])
plt.show()

# Split up data into training and validation set
X_iris = iris.drop('species', axis=1)  # extract the independent data
print(X_iris.shape)

y_iris = iris['species']  # extract the categorical data
print(y_iris.shape)

# Initialize and use dimensionality
pcaModel = PCA(n_components=2)  # initialize PCA model to reduce dimensionality from 4 to 2
pcaModel.fit(X_iris)  # fit model to the higher dimensioned data
X_2dimension = pcaModel.transform(X_iris)  # transform the data to 2 dimensions

# Augment current dataframe `iris` with new data from PCA transformation
iris['PCA1'] = X_2dimension[:, 0]
iris['PCA2'] = X_2dimension[:, 1]

# Initialize and train classification model
gmmModel = GaussianMixture(n_components=3, covariance_type='full')  # initialize GMM model
gmmModel.fit(X_iris)  # fit model to the model
y_predicted = gmmModel.predict(X_iris)  # predict what the cluster labels are

# Augment current dataframe `iris` with new data from GMM prediction
iris['cluster'] = y_predicted

# Plot resulting data set
sns.lmplot("PCA1", "PCA2", hue="species", col='cluster', scatter_kws={"s": 6}, data=iris, fit_reg=False)
plt.show()

from sklearn import neighbors, datasets
from matplotlib.colors import ListedColormap
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print(__doc__)

breast_df: pd.DataFrame = pd.read_csv(
    "./data/breast-cancer-wisconsin-data/breast_cancer.csv"
)

n_neighbors = 25

# import some data to play with
iris = datasets.load_iris()

# we only take the first two features. We could avoid this ugly
# slicing by using a two-dim dataset
X = iris.data[:, :2]
print(X.shape)
y = iris.target

h = 0.02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(["orange", "cyan", "cornflowerblue"])
cmap_bold = ["darkorange", "c", "darkblue"]

for weights in ["uniform"]:  # ["uniform", "distance"]
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    print(np.arange(x_min, x_max, h))

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    print(f"xx and yy shape: {xx.shape, yy.shape}")

    plt.plot(xx, yy, marker=".", color="k", linestyle="none")

    print(
        f"np.c_[xx.ravel(), yy.ravel()] shape is {np.c_[xx.ravel(), yy.ravel()].shape}"
    )
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    print(f"predictions: {Z}, shape of predictions: {Z.shape}")

    # Put the result into a color plot
    print(xx.shape)
    Z = Z.reshape(xx.shape)
    print(Z)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    sns.scatterplot(
        x=X[:, 0],
        y=X[:, 1],
        hue=iris.target_names[y],
        palette=cmap_bold,
        alpha=1.0,
        edgecolor="black",
    )
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(
        "3-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights)
    )
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])

plt.show()
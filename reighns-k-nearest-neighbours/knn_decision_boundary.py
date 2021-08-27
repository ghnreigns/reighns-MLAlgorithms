from sklearn import neighbors, datasets
from matplotlib.colors import ListedColormap
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""
For each point in the mesh grid, we perform a prediction, and plot it.
As h-> 0, the predictions => infinity, and can therefore form a decision boundary.
So xx and yy are combination of 2 features (hypothetically) generated from the min of x1 x2 and max x1 x2
Then we take the cross product combination of xx and yy to "make predictions".

In our simplfied example, 
"""

clf_data = np.array(
    [
        [-1, 1, 0],
        [1, -1, 0],
        [0, 2, 0],
        [2, 2, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 2, 1],
    ]
)

X = clf_data[:, 0:2].reshape(-1, 2)
print(X)
y = clf_data[:, 2]
y_test = np.array([[1, 1]])


n_neighbors = 1
clf = neighbors.KNeighborsClassifier(n_neighbors, weights="uniform")
clf.fit(X, y)
preds = clf.predict(y_test)
print(preds)

h = 0.5  # step size in the mesh
# Create color maps
cmap_light = ListedColormap(["orange", "cyan"])
cmap_bold = ["r", "g"]

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
print(np.arange(x_min, x_max, h))

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
print(f"xx and yy: {xx, yy}")
print(f"xx and yy shape: {xx.shape, yy.shape}")

mesh_grid = plt.plot(xx, yy, marker=".", color="k", linestyle="none")

print(f"np.c_[xx.ravel(), yy.ravel()] subset is {np.c_[xx.ravel(), yy.ravel()]}")
print(f"np.c_[xx.ravel(), yy.ravel()] shape is {np.c_[xx.ravel(), yy.ravel()].shape}")

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
print(f"predictions: {Z}, shape of predictions: {Z.shape}")

# Put the result into a color plot
print(xx.shape)
Z = Z.reshape(xx.shape)
print(Z)
contour_decision_boundary = plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light)
# fig_contour = contour_decision_boundary.get_figure()
# fig_contour.savefig('contour_decision_boundary.png', dpi=400)

# Plot also the training points
for k_neighbours in range(1, 8, 2):
    data_plot = sns.scatterplot(
        x=X[:, 0],
        y=X[:, 1],
        hue=y,
        palette=cmap_bold,
        alpha=1.0,
        edgecolor="black",
    )
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(
        "2-Class classification (k = %i, weights = '%s')" % (k_neighbours, "uniform")
    )
    # plt.xlabel(iris.feature_names[0])
    # plt.ylabel(iris.feature_names[1])

    # figure = data_plot.get_figure()
    # figure.savefig(f'knn_plot_{k_neighbours}.png', dpi=400)
    # plt.clf()
    plt.show()
    break
import importlib
import logging
import os
import pprint
import sys  # noqa

import numpy as np

# import pandas as pd

sys.path.append(os.getcwd())  # noqa
Entropy = importlib.import_module("reighns-utils.scripts.entropy", package="reighns-utils")

logging.basicConfig(filename="example.log", filemode="w", level=logging.DEBUG)


"""
Decision Trees are greedy algorithms
that maximise the current Information Gain
without backtracking or going back up to the root.
Future splits are based on the current splits:
split(t+1) = f(split(t))
At every level, the impurity of the dataset
decreases. The entropy (randomness) decreases
with the level.
"""


class DTNode:
    def __init__(
        self,
        feat_idx=None,
        threshold=None,
        left=None,
        right=None,
        info_gain=None,
        class_label=None,
    ):
        self.feat_idx = feat_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.class_label = class_label


class DecisionTreeClassifier:
    def __init__(self, max_depth=2, min_samples_split=2):
        # min_samples_split means min_samples_split in scikit-learn
        # max_depth equals max_depthint, default=None
        # The maximum max_depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def build_tree(self, dataset, cur_depth=0):
        # take our x_train and y_train
        x, y = dataset[:, :-1], dataset[:, -1]

        # num_sample and num_features, 149 samples, 4 features
        num_sample, num_feature = x.shape

        assert x.shape == (num_sample, num_feature), y.shape == (num_sample,)

        # recursively build the subtrees

        if num_sample >= self.min_samples_split and cur_depth <= self.max_depth:

            # best split
            best_split = self.get_best_split(dataset, num_sample, num_feature)

            # print(best_split)

            my_complex_dict = pprint.pformat(best_split)

            # logging.info(f"My complex dict:\num_sample{my_complex_dict}")

            if best_split["info_gain"] > 0:
                left_tree = self.build_tree(best_split["left"], cur_depth + 1)
                right_tree = self.build_tree(best_split["right"], cur_depth + 1)

                return DTNode(
                    best_split["feat_idx"],
                    best_split["threshold"],
                    left_tree,
                    right_tree,
                    best_split["info_gain"],
                )

        y = list(y)
        class_label = max(y, key=y.count)  # class label = majority count at leaves

        return DTNode(class_label=class_label)

    def get_best_split(self, dataset, num_sample, num_feature):

        # for each feature, for each unique value that each feature can take, we loop through and find the "best" attribute
        # that has the highest info gain

        best_split = {}
        # initiate info gain as -infinity
        max_info_gain = -float("inf")

        # for each feature x_i
        for idx in range(0, num_feature):
            # denote feat_val as the column vector $X_{i}$ where $X$ is the data matrix.
            # note that we are using all rows but subsetting on the feature column
            feat_val = dataset[:, idx]

            # Find all possible values that this feature $x_{i}$ can take on.
            # Denote possible_boundss as A_{x_{i}} = {a_i} where a_i denotes the unique value that x_i can take on.
            possible_boundss = np.unique(feat_val)

            # for each unique value in the set A_{x_{i}}
            for thresh in possible_boundss:
                # for each row of training data of tuple (x^{(i)}, y^{(i)}), we check if this row's feature i is smaller or bigger than threshold.
                # in first loop, we check feature x_{1} so for each row, we just check the first element.
                # and split accordignly to two brances.
                data_left = np.array([row for row in dataset if row[idx] <= thresh])
                data_right = np.array([row for row in dataset if row[idx] > thresh])

                # ensure that
                if len(data_left) > 0 and len(data_right) > 0:
                    # basically: this step calculates info gain of each sub-dataset and chooses the best one.

                    y, left_y, right_y = (
                        dataset[:, -1],
                        data_left[:, -1],
                        data_right[:, -1],
                    )
                    cur_info_gain = self.get_info_gain(y, left_y, right_y)

                    if cur_info_gain > max_info_gain:
                        best_split["feat_idx"] = idx
                        best_split["threshold"] = thresh
                        best_split["left"] = data_left
                        best_split["right"] = data_right
                        best_split["info_gain"] = cur_info_gain
                        max_info_gain = cur_info_gain

        tree_counter: int = 1  # to keep track number of splits
        print(f"Tree Split : {tree_counter}")
        tree_counter += 1

        return best_split

    def get_info_gain(self, parent, left, right):
        # H(A_{x_i}) = \dfrac{num of elements in A_{x_i, a_i}}{num of A_{x_i}}

        assert len(parent) == len(left) + len(right)
        print(
            f"Num of elements in parent is {len(parent)} \n in left is {len(left)} \n in right is {len(right)}"
        )
        weight_left = len(left) / len(parent)
        weight_right = len(right) / len(parent)

        info_gain = self.get_entropy(parent) - (
            weight_left * self.get_entropy(left) + weight_right * self.get_entropy(right)
        )

        return info_gain

    def get_entropy(self, y):
        # Get entropy of a dataset, in particular, get the entropy of the y_true list.
        # For details go utils and see.
        # In our tree code, expect to calculate it in every decision.
        return Entropy.calculate_entropy(y)

    def fit(self, x, y):
        dataset = np.concatenate((x, y), axis=1)
        self.root = self.build_tree(dataset)

    def make_pred(self, x, root):
        if root.class_label is not None:
            return root.class_label

        feat_val = x[root.feat_idx]

        if feat_val <= root.threshold:
            return self.make_pred(x, root.left)
        else:
            return self.make_pred(x, root.right)

    def predict(self, x):
        return [self.make_pred(i, self.root) for i in x]


if __name__ == "__main__":

    import random
    from random import shuffle

    from sklearn import tree
    from sklearn.datasets import load_iris
    from sklearn.metrics import accuracy_score

    random.seed(1992)

    iris = load_iris()
    ATTRIBUTE_MAP = {
        "x_0": "sepal length (cm)",
        "x_1": "sepal width (cm)",
        "x_2": "petal length (cm)",
        "x_3": "petal width (cm)",
    }
    CLASS_MAP = {0.0: "setosa", 1.0: "versicolor", 2.0: "virginica"}

    X, y = iris.data, iris.target
    shuffle(X), shuffle(y)
    X, y = X[:20, :], y[:20]
    y = y.reshape(-1, 1)  # reshape to concat in code.
    print(X.shape, y.shape)
    # logging.info(f"My complex dict:\num_sample{X}")
    # logging.info(f"My complex dict:\num_sample{y}")

    # clf = tree.DecisionTreeClassifier()
    # clf = clf.fit(X, y)

    # cols = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
    # df = pd.read_csv(
    #     "/home/reighns/reighns-MLAlgorithms/reighns-decision-trees/data/iris.csv",
    #     skiprows=1,
    #     header=0,
    #     names=cols,
    # )

    # # replace class strings with integer indices
    # df["class"] = df["class"].str.replace("Iris-setosa", "0")
    # df["class"] = df["class"].str.replace("Iris-versicolor", "1")
    # df["class"] = df["class"].str.replace("Iris-virginica", "2")
    # df["class"] = df["class"].map(lambda x: int(x))

    # X = df.iloc[:, :-1].values
    # Y = df.iloc[:, -1].values.reshape(-1, 1)
    # X = np.array(X)
    # Y = np.array(Y)

    # print(X.shape, Y.shape)

    clf = DecisionTreeClassifier()
    clf.fit(X, y)  # split this into training and testing datasets
    pred = clf.predict(X)
    print(accuracy_score(y, pred))
    # print(pred)

    def print_tree(root=None, indent="  "):
        if root.class_label is not None:
            class_ = root.class_label
            class_map = CLASS_MAP[class_]
            print(f"class {int(class_)} - {class_map}")
        else:
            attribute = "x_" + str(root.feat_idx)
            mapped_attribute = ATTRIBUTE_MAP[attribute]
            print(
                mapped_attribute,
                "<=",
                root.threshold,
                ":",
                format(root.info_gain, "0.4f"),
            )
            # print()
            print(indent + "left: ", end="")
            print_tree(root.left, indent + indent)
            print(indent + "right: ", end="")
            print_tree(root.right, indent + indent)

    print_tree(clf.root)

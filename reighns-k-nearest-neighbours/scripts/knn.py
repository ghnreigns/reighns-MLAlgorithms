import os
import sys  # noqa

sys.path.append(os.getcwd())  # noqa

import importlib

DistanceMetrics = importlib.import_module(
    "reighns-distance-metrics.distance", package="reighns-distance-metrics"
)


import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


class reighnsKNN:
    def __init__(
        self,
        k: int,
        distance_metric: DistanceMetrics = DistanceMetrics.euclidean_distance,
        mode: str = "classification",
    ):
        r"""KNN Algorithm from Scratch.

        # Introduction

        K-Nearest Neighbors is one of the most basic yet essential classification algorithms in Machine Learning. It belongs to the supervised learning domain and finds intense application in pattern recognition, data mining and intrusion detection. It is widely disposable in real-life scenarios since it is non-parametric, meaning, it does not make any underlying assumptions about the distribution of data (as opposed to other algorithms such as GMM, which assume a Gaussian distribution of the given data).

        ---

        # Notations and Definitions

        - xxx

        ---

        # Hypothesis Space and the Learning Algorithm

        The k-NN regression is a nonparametric approach. The hypothesis space is the class of all functions that are *piecewise constant* on the cells of the $k$th order Voronoi diagram of some set of $n$ points in $\mathbb R^d$.

        To be more precise, for a collection of points $\{x_i\}_{i=1}^n \subset \mathbb R^d$, let $V_j^k(\{x_i\}_{i=1}^n)$ be the $j$th Voronoi cell in the $k$th order Voronoi partition of the space by points $\{x_i\}_{i=1}^n$ (let us say we order these cells in some way in order to index them). This means that all the points in each of these cells have the same $k$-nearest neighbors among $\{x_i\}_{i=1}^n$. Then, the class of functions underlying $k$-NN can be written as

        \begin{align}
        \mathcal F_{\text{$k$-NN}}^{(n)} = \Big\{f: \mathbb R^d \to \mathbb R:&\; \text{There exists $\{x_i\}_{i=1}^n \subset \mathbb R^d$ and $\{a_j\} \subset \mathbb R$ such that} \\ &f(x) = \sum_j a_j \cdot 1_{V_j^k(\{x_i\}_{i=1}^n)}(x), \quad \forall x \in \mathbb R^d. \Big\}.
        \end{align}

        Here, $1_{A}(x) = 1\{x \in A\}$ is the indicator function of set $A$. As you can see, this is a rather large class of functions. There are many ways to pick $n$ points in $\mathbb R^d$ and each one defines a potentially different Voronoi partition. If you consider all such partitions and all functions that are constant over cells of those partitions, you would get the $k$-NN class.

        You can also take the union of $F_{\text{k-NN}}^{(n)}$ over all $n \in \{1,2,\dots\}$ to define all possible $k$-NN functions.


        # Algorithm

        For this and time/space complexity, the code we reference is in `knn_naive.py`.


        Formally, given the following:

        - Data matrix: $\mathcal{X}_{m \times n}$ where any element $x \in \mathcal{X}$ is in $\mathbb{R}^{n}$.
        - y_true: $\mathcal{y}_{m \times n}$
        - A positive integer K,
        - A test array $\mathcal{X}_{test}$ where any unseen observation $x_{t} \in X_{test}$ are of dimension $\mathbb{R}^{n}$,
        - A distance metric $\mathcal{d}$ used to calculate distance of two points,

        KNN classifier performs the following steps (for simplicity sake, we use just one $x_{t}$ and referencing to the more naive function `KNN_example`):

        1. Initiate a list `distance_list = []` which stores a tuple consisting of (distance, class).
        2. Perform a distance metric, say Euclidean Distance, on $x_{t}$ and each point $x_{i} \in \mathcal{X}$. Mathematically, let $\mathcal{d}(x_{i}, x_{t})$ be the distance of point $x_{i}$ and $x_{t}$. Let $y_{i}$ be the corresponding class of $x_{i}$.
        3. Append the tuple $(\mathcal{d}(x_{i}, x_{t}), y_{i})$ into `distance_list`.
        4. Sort `distance_list` on the first element, that is, to sort it according to $\mathcal{d}(x_{i}, x_{t})$ in ascending order.
        5. Find the top K elements in the sorted list, name this list `top_k`.
        6. Find the mode of the classes in `top_k`. That is to say, perform a majority vote to see in this `top_k` list, which class appears the most frequent.
        7. Assign the majority class to $x_{t}$, and we have done the prediction of $x_{t}$. Note that this is just one point.

        ---

        # Time and Space Complexity

        ## Time Complexity

        First note that Euclidean Distance between two points of dimension $\mathbb{R}^{n}$ is of $\mathcal{O}(n)$. This is because the raw form of Euclidean Distance is just one single for loop, performing $n$ subtractions, then $n$ square and lastly $n-1$ additions with one square root operation at the end. This totals up to $(n+n+(n-1)+1)=3n$ which is just $\mathcal{O}(n)$.

        Assuming $\mathcal{X}$ and `y_true` are matrices of $m \times n$ and $m \times 1$ dimension respectively.

        1. Then the first for loop in the function `KNN_example` is simply $\mathcal{O}(m \cdot n)$ because in each loop, we take a total of $\mathcal{O}(n)$ time as shown earlier in Euclidean Distance, since there is $m$ loops, the total time follows.
        2. Sorting the `distance_list` using `sorted` takes $\mathcal{O}(m \cdot \log m)$ - this is a well known time complexity in Python.
        3. Calculating the majority class in our code takes $\mathcal{O}(k)$ time.
        4. It totals up to $\mathcal{O}(m \cdot n + m \cdot \log m + k)$. And since $k$ in our algorithm is usually small relative to $m$ and $n$, we can therefore remove it to become $\mathcal{O}(m \cdot n + m \cdot \log m)$.
        5. Lastly, since we may be feeding in more than one test sample, and assuming there exists $p$ test samples. Then our final time complexity is $\mathcal{O}(p \cdot m \cdot n + p \cdot m \cdot \log m)$.
        ---

        ## Space Complexity

        It is worth noting that in a general DSA course, we do not consider input as space, but in ML, this may be different. An intuition is if your input data is too memory consuming, your model cannot even train propery. It will throw Memory Limit Exceeded Error!

        1. Input Matrix $\mathcal{X}$ is a list/array which takes up space $\mathcal{O}(m \cdot n)$.
        2. Distance List `distance_list` takes up roughly $\mathcal{O}(m \cdot n)$.
        3. Total Space Complexity: $\mathcal{O}(2 \cdot m \cdot n) \rightarrow \mathcal{O}(m \cdot n)$.

        ## Improvements

        Looking at the above code the major bottleneck is the sorting function. We can reduce this by using the quickselect method is which finds the nth smallest element and partitions the array into those smaller than this element and those larger. We can use this method once to get the the k nearest neighbors from a list. The average time complexity of this method is $\mathcal{O}(m)$ where m is the length of the list on average.

        - Time Complexity: ​Since the sorting method becomes $\mathcal{O}(m)$ then the time complexity become $\mathcal{O}(m \times n + m)$ which can be reduced to $\mathcal{O}(m \times n)$.

        - Space Complexity: Same. We did not change the space complexity since the quickselect method operates directly on our distances list.

        See `knn_quick_select.py` for the code.

        ---

        # Decision Boundary and K

        ## K = 1

        It is worth noting that ***Training Error*** is 0 when $K=1$. In ML, this just means that you input back your training set in the algorithm and check the error rate (or accuracy rate for this matter). Now this is exactly because when $K$ is 1, and say a point $x$ from the training set $\mathcal{X}$ is $(1,2)$, then its nearest neighbour is just itself, as any distance metric $\mathcal{d}(x,x) = 0$.

        However, ***Test Time Error*** won't be so good simply because when $K=1$, the variance of the model predictions is large, and may not generalize well.

        ## K = m

        K = m implies something worse, because there exists m data points, if $K = m$, and assuming number of positive is more than the number of negatives (vice-versa), then the whole set of $\mathcal{D}$ will be the decision boundary and any prediction of the query point $x_{q}$ will be assigned to the positive class/label no matter if the true label is positive or negative!

        ## K's impact on Bias and Variance

        ### Small K

        A small value for K provides the most flexible fit, and in ML terminology, flexible usually means "more complex/complicated" model which has a low bias but high variance. In the decision boundary (see image on Decision Boundary for K=1), you will see the edges are more jagged, as the classifier KNN is trying hard to fit (overfit) to the training data. This may result in the model not able to generalize as you present a different training set to KNN.

        ### Big K

        In contrast, a big value for K, will remove outliers that otherwise a small K won't. Hence it is more robust to outliers, and hence lower variance but higher bias.

        ## How to determine K?

        This is a popular interview question when talking about k-NN. We should be familiar with at least two approaches

        Two popular methods are:

        - k = sqrt(total number of training data points) and round down or up to an odd number.

        - (See notebook) We can create a plot between accuracy (or loss) and K via cross-validation Then we choose either the K which produces the highest accuracy or the lowest loss.

        ---

        # Normalization

        We need to normalize our data before performing KNN. The reasons are as follows, and for visualization see notebook's Normalization Section.

        Suppose you had a dataset (m "examples" by n "features") and all but one feature dimension had values strictly between 0 and 1, while a single feature dimension had values that range from -1000000 to 1000000. When taking the euclidean distance between pairs of "examples", the values of the feature dimensions that range between 0 and 1 may become uninformative and the algorithm would essentially rely on the single dimension whose values are substantially larger. Just work out some example euclidean distance calculations and you can understand how the scale affects the nearest neighbor computation.

        From Introduction to Statistical Learning, it is also mentioned that:

        Because the KNN classifier predicts the class of a given test observation by identifying the observations that are nearest to it, the scale of the variables matters. Variables that are on a large scale will have a much larger effect on the distance between the observations, and hence on the KNN classifier, than variables that are on a small scale. For instance, imagine a data set that contains two variables, salary and age (measured in dollars and years, respectively). As far as KNN is concerned, a difference of $1,000 in salary is enormous compared to a difference of 50 years in age. Consequently, salary will drive the KNN classification results, and age will have almost
        no effect. This is contrary to our intuition that a salary difference of $1,000 is quite small compared to an age difference of 50 years. Furthermore, the importance of scale to the KNN classifier leads to another issue: if we measured salary in Japanese yen, or if we measured age in minutes, then we’d get quite different classification results from what we get if these two variables are measured in dollars and years.

        ---

        # Summary

        This section aptly summarizes the KNN Algorithm in a nutshell. And should serve as flashcard questions during revision using the Active Recall methodology.

        - KNN can be used for both regression and classification problems. See my drawing named "classification_vs_regression".
            - Classification takes the mode of the top K nearest labels.
            - Regression takes the mean of the top K nearest labels.

        - KNN is **non-parametric** and **instance-based** **supervised** learning algorithm.
            - Non-parametric: means it makes no explicit assumptions about the functional form of h, avoiding the dangers of mismodeling the underlying distribution of the data. For example, suppose our data is highly non-Gaussian but the learning model we choose assumes a Gaussian form. In that case, our algorithm would make extremely poor predictions.
            - Instance-based: learning means that our algorithm doesn’t explicitly learn a model. Instead, it chooses to memorize the training instances which are subsequently used as “knowledge” for the prediction phase. Concretely, this means that only when a query to our database is made (i.e. when we ask it to predict a label given an input), will the algorithm use the training instances to spit out an answer.

        - For a complete algorithm breakdown please refer to the Algorithm section.

        - Undesirable consequences ensues since classifying just one single query data will require the KNN to run through the whole training data. Each time you query a new data point, we need to run the "training" phase. This is in contrast of other models like Neural Network whereby once you trained a model with the weights, you can use the weights to inference the query point.

        - Religiously, I mention the definition of Bias and Variance in every algorithm. In this case, please read [Wikipedia](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff) section on KNN for formal mathematical definition.
            - Bias: The bias error is an error from erroneous assumptions in the learning algorithm. High bias can cause an algorithm to miss the relevant relations between features and target outputs (underfitting).
            - Variance: The variance is an error from sensitivity to small fluctuations in the training set. High variance may result from an algorithm modeling the random noise in the training data (overfitting).

        - How to choose the hyperparameter K? Read the section on Decision Boundary and K.

        - The usual hyperparameters of a KNN classifier are the **distance metric** $\mathcal{d}$ and our $K$.

        -  We know that k-NN can be computationally expensive when the dataset is large. How can you alleviate this?
            - We need to think about what the question is asking. It is asking how can we reduce the training time for a k-NN. One method we know to reduce training time for any model is dimensionality reduction using methods such as PCA, LDA etc. Another method could be that we randomly sample the data to reduce the training data size, however, this increases the variance of our model so we need to increase k to offset this. Lastly, we can cluster points within a region and treat all these data points as a single point. The coordinates of this single point will be the centre of all the data points. A strong candidate would be able to give the dimensionality reduction and at least one other method as the answer.

        - How can we use k-NN for missing values in datasets?
            - We can simply impute a value by getting the average (or majority class) of the closest k neighbors. We can use Euclidean distance as our distance metric. This is very useful and we should be aware of this.


        - Some properties of KNN worth noting:
            - More data implies better KNN, intuitively, this is true for many algorithms, but why KNN in particular?
            - K is usually chosen to be odd to avoid ties in majority voting.
            - Small K has complex and jagged surface while big K smoothes the decision surface, removing outliers.
            - K = 1 and K = m edge cases: Read the section on Decision Boundary and K.

        - Normalization before training (see Normalization).

        ---

        # References

        - [kevinzakka's end-to-end KNN](https://kevinzakka.github.io/2016/07/13/k-nearest-neighbor/): Primarily based off this reference.
        - [KNN Regression](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761)
        - [Time and Space Complexity](https://towardsdatascience.com/k-nearest-neighbors-computational-complexity-502d2c440d5)
        - [Hypothesis Space](https://stats.stackexchange.com/questions/200766/hypothesis-space-of-naive-bayes-and-knn)
        - [Steps to plot Decision Boundary](https://stats.stackexchange.com/questions/370531/knn-decision-boundary)
        - [KNN is discriminative learning](https://stats.stackexchange.com/questions/105979/is-knn-a-discriminative-learning-algorithm/106081#106081)
        - [Video on Decision Boundary of KNN](https://youtu.be/k_7gMp5wh5A)
        - [Important on Bias Variance](http://scott.fortmann-roe.com/docs/BiasVariance.html)
        - [Intuition on Scaling Data](https://stats.stackexchange.com/questions/287425/why-do-you-need-to-scale-data-in-knn)
        - [Intuition on weighted KNN](https://www.geeksforgeeks.org/weighted-k-nn/)

        Args:
            k (int): K in K-Nearest-Neighbours
            distance_metric (str, optional): [description]. Defaults to 'euclidean_distance'.
        """
        self.k: int = k
        self.distance_metric = distance_metric
        self.mode = mode

    def _vote(self, neighbor_labels):
        """Return the most common class among the neighbor samples"""
        frequency_counts = np.bincount(neighbor_labels.astype("int"))
        majority_vote = np.argmax(frequency_counts)
        return majority_vote

    def _mean(self, neighbor_labels):
        """Average the top K nearest neighbours in Regression"""
        return sum(neighbor_labels) / len(neighbor_labels)

    def check_shape(self, X: np.ndarray = None, y: np.ndarray = None) -> None:
        """Always call `np.asarray()` on the incoming inputs as this is a good way to check whether the user input funny data types. And it also conveniently turns list into nparray.

        Args:
            X (np.ndarray, optional): [description]. Defaults to None.
            y (np.ndarray, optional): [description]. Defaults to None.
        """
        if X is not None:
            X = np.asarray(X)
            assert len(X.shape) == 2, "The input X matrix should be a 2-d array!"

        if y is not None:
            assert len(y.shape) == 1, "Both the y_true and y_pred array should be a 1-d array!"

    def predict(self, X_train: np.ndarray, y_true: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        """The following steps details one cycle of classification. The regression follows just by changing the mode of the init.

        line 1-2: Check shape of input arrays.

        line 3: Initialize an empty y_pred array to be same size of X_test. This is a 1d-array of predictions in classes (int) which corresponds to X_test.

        line 4 - end: We explain just the first loop, and then the subsequent loops is the same.

            1. [self.distance_metric(test_sample, x) for x in X_train]: take the first unclassified sample x_q in X_test, this code computes the distance of x_q with ALL of X_train data and store in a list.
               Example: test = [1,2,3], train = [[1,1,1], [2,2,2]] we then compute distance of test with each x_i of X_train. Final output is a 1d array [0.1, 0.3, ...]

            2. np.argsort([self.distance_metric(test_sample, x) for x in X_train]): applied np.argsort on point 1 where we sort the list in 1. This array returns the index of the sorted array. This is important to know.
               Example: x = np.array([30, 10, 20]) -> np.argsort(x) -> array([1, 2, 0])
               Note this is sorted according to index because argsort first sort it according to ascending order, which should be [10, 20, 30].
               However, the function returns [1, 2, 0] because they know the smallest number is 10, and it resides at index 1, second number is 20, residing at index 2, etc.

            3. np.argsort([self.distance_metric(test_sample, x) for x in X_train])[:self.k]: Simple slicing on point 2 to get the top k nearest neighbours' index.

            4. k_nearest_neighbors_classes = np.array([y_true[i] for i in k_nearest_neighbors_idx]): This simply finds the corresponding true labels corresponding to the indexes found in 3.

            5. y_pred[i] = self._vote(k_nearest_neighbors_classes): Now, call _vote on this array, simply put, if this array is [1,1,2,3,1,1,3], then by majority vote, class 1 is the chosen prediction for this test sample x_q.


        Args:
            X_train (np.ndarray): [description]
            y_true (np.ndarray): [description]
            X_test (np.ndarray): [description]

        Returns:
            np.ndarray: [description]
        """
        self.check_shape(X=X_train, y=y_true)
        self.check_shape(X=X_test, y=None)

        y_pred = np.empty(X_test.shape[0])

        for i, test_sample in enumerate(X_test):

            k_nearest_neighbors_idx = np.argsort(
                [self.distance_metric(test_sample, x) for x in X_train]
            )[: self.k]

            k_nearest_neighbors_classes = np.array([y_true[i] for i in k_nearest_neighbors_idx])
            if self.mode == "classification":
                y_pred[i] = self._vote(k_nearest_neighbors_classes)

            elif self.mode == "regression":
                y_pred[i] = self._mean(k_nearest_neighbors_classes)

        return y_pred

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Return probability estimates for the test data X.

        Only Applicable for Classification

        Args:
            X_test (np.ndarray): [description]

        Returns:
            p: np.ndarray: [description]
        """


if __name__ == "__main__":
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.7, random_state=42
    )

    sklearn_knn = KNeighborsClassifier(n_neighbors=3)
    sklearn_predictions = sklearn_knn.fit(X_train, y_train).score(X_test, y_test)

    HN_KNN_CLASSIFICATION = reighnsKNN(
        k=3, distance_metric=DistanceMetrics.euclidean_distance, mode="classification"
    )
    HN_CLASSIFICATION_PREDICTIONS = HN_KNN_CLASSIFICATION.predict(X_train, y_train, X_test)

    print(HN_CLASSIFICATION_PREDICTIONS)
    print("\nSKLEARN Accuracy score : %.3f" % (sklearn_predictions * 100))
    print(
        "\nHN Accuracy score : %.3f" % (accuracy_score(y_test, HN_CLASSIFICATION_PREDICTIONS) * 100)
    )
    print()

    # print(sklearn_knn.fit(X_train, y_train).predict_proba(X_test))
    # print("Recall score : %f" % (sklearn.metrics.recall_score(y_val, preds) * 100))
    # print("ROC score : %f\n" % (sklearn.metrics.roc_auc_score(y_val, preds) * 100))
    # print(sklearn.metrics.confusion_matrix(y_val, preds))

    X = np.array([[0], [1], [2], [3]])

    y = np.array([0, 0, 1, 1])
    from sklearn.neighbors import KNeighborsRegressor

    neigh = KNeighborsRegressor(n_neighbors=2)
    neigh.fit(X, y)

    print(neigh.predict([[1.5]]))
    HN_KNN_REGRESSION = reighnsKNN(
        k=2, distance_metric=DistanceMetrics.euclidean_distance, mode="regression"
    )
    HN_REGRESSION_PREDICTIONS = HN_KNN_REGRESSION.predict(X, y, np.array([[1.5]]))

    print(HN_REGRESSION_PREDICTIONS)

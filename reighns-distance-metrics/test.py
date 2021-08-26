import numpy as np
import scipy

import distance as d

### Euclidean Distance Testing ###
test_distance = np.array([[-4, -3, -2], [-1, 0, 1]])

print(np.linalg.norm(test_distance[0] - test_distance[1]))
print(np.linalg.norm(test_distance[0] - test_distance[1]).reshape(-1, 1))
print(d.euclidean_distance(test_distance[0], test_distance[1]))
print(scipy.spatial.distance.euclidean(test_distance[0], test_distance[1]))

print(d.manhattan_distance(test_distance[0], test_distance[1]))
print(scipy.spatial.distance.minkowski(test_distance[0], test_distance[1], p=1))


print(scipy.spatial.distance.cosine(test_distance[0], test_distance[1], w=None))
print(d.cosine_distance(test_distance[0], test_distance[1]))


### CS3244 ###

x1 = np.asarray([1, 2, 3])
x2 = np.asarray([0, 0, 0])
print(
    "Euclidean Distance between (1, 2, 3) and (0, 0, 0):", d.euclidean_distance(x1, x2)
)
print(
    "Manhattan Distance between (1, 2, 3) and (0, 0, 0):", d.manhattan_distance(x1, x2)
)

x1 = np.asarray([100, 20, 30])
x2 = np.asarray([0, 0, 0])
print(
    "Euclidean Distance between (100, 20, 30) and (0, 0, 0):",
    d.euclidean_distance(x1, x2),
)
print(
    "Manhattan Distance between (100, 20, 30) and (0, 0, 0):",
    d.manhattan_distance(x1, x2),
)

import numpy as np
import pytest
from typing import Union

"""
Euclidean distance
There are many ways to measure the distance between two vectors. Let's write a function that computes the Euclidean distance between vectors.

This function should take as input two vectors and return the euclidean distance between them.
"""


def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute the Euclidean distance between two vectors.

    Parameters
    ----------
    v1 : np.ndarray
        First vector.
    v2 : np.ndarray
        Second vector.

    Returns
    -------
    float
        Euclidean distance between `v1` and `v2`.
    """
    dist = v1 - v2
    return np.linalg.norm(dist, axis=len(dist.shape) - 1)


v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
dist = np.sqrt(np.sum((v2 - v1) ** 2))
assert euclidean_distance(v1, v2) == pytest.approx(dist)

"""
KNN search
Using the distance function you just wrote, write a function that finds the k-nearest neighbors of a query vector.

This function should take as input a query vector, a 2d array of database vectors, and an integer k the number of 
nearest neighbors to return. And it should return the vectors that are the k-nearest neighbors of the query vector.
"""


def find_nearest_neighbors(query: np.ndarray,
                           vectors: np.ndarray,
                           k: int = 1) -> np.ndarray:
    """
    Find k-nearest neighbors of a query vector.

    Parameters
    ----------
    query : np.ndarray
        Query vector.
    vectors : np.ndarray
        Vectors to search.
    k : int, optional
        Number of nearest neighbors to return, by default 1.

    Returns
    -------
    np.ndarray
        The `k` nearest neighbors of `query` in `vectors`.
    """
    distances = euclidean_distance(query, vectors)
    indices = np.argsort(distances)[:k]
    return vectors[indices, :]


mat = np.random.randn(1000, 32)
query = np.random.randn(32)
k = 10
norms = np.linalg.norm(mat, axis=1)
expected = np.linalg.norm(mat - query, axis=1)
expected = mat[np.argsort(expected)[:k], :]
actual = find_nearest_neighbors(query, mat, k=k)
assert np.allclose(actual, expected)

"""
Other distance metrics
For this problem we'll write a new distance function and modify our nearest neighbors function to accept a distance metric.

Write a function that computes the cosine distance between vectors.
"""


def cosine_distance(v1: np.ndarray, v2: np.ndarray) -> Union[float, np.ndarray]:
    """
    Compute the cosine distance between two vectors.

    Parameters
    ----------
    v1 : np.ndarray
        First vector.
    v2 : np.ndarray
        Second vector.

    Returns
    -------
    float
        Cosine distance between `v1` and `v2`.
    """
    vecs = (v1, v2) if len(v1.shape) >= len(v2.shape) else (v2, v1)
    return 1 - np.dot(*vecs) / (
            np.linalg.norm(v1, axis=len(v1.shape) - 1) *
            np.linalg.norm(v2, axis=len(v2.shape) - 1)
    )


v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
v1_norm = np.linalg.norm(v1)
v2_norm = np.linalg.norm(v2)
dist = 1 - np.dot(v2, v1) / (v1_norm * v2_norm)
assert cosine_distance(v1, v2) == pytest.approx(dist)


# Now, rewrite the find_nearest_neighbors function to accept a distance metric so you can use either Euclidean or Cosine distance


def find_nearest_neighbors_distance_metric(query: np.ndarray,
                                           vectors: np.ndarray,
                                           k: int = 1,
                                           distance_metric="euclidean") -> np.ndarray:
    """
    Find k-nearest neighbors of a query vector with a configurable
    distance metric.

    Parameters
    ----------
    query : np.ndarray
        Query vector.
    vectors : np.ndarray
        Vectors to search.
    k : int, optional
        Number of nearest neighbors to return, by default 1.
    distance_metric : str, optional
        Distance metric to use, by default "euclidean".

    Returns
    -------
    np.ndarray
        The `k` nearest neighbors of `query` in `vectors`.
    """
    if distance_metric == "euclidean":
        distances = euclidean_distance(query, vectors)
    elif distance_metric == "cosine":
        distances = cosine_distance(query, vectors)
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")
    indices = np.argsort(distances)[:k]
    return vectors[indices, :]


mat = np.random.randn(1000, 32)
query = np.random.randn(32)
k = 10
norms = np.linalg.norm(mat, axis=1)
for dist in ["euclidean", "cosine"]:
    if dist == "euclidean":
        expected = np.linalg.norm(mat - query, axis=1)
    else:
        expected = 1 - np.dot(mat, query) / (norms * np.linalg.norm(query))
    expected = mat[np.argsort(expected)[:k], :]
    actual = find_nearest_neighbors_distance_metric(query, mat, k=k, distance_metric=dist)
    assert np.allclose(actual, expected)

"""
Exploration
Now that we have a nearest neighbors function that accepts a distance metric,
let's explore the differences between Euclidean distance and cosine distance.

Would you expect same or different answers?

Note that if you normalize the vectors, then Euclidean distance and cosine distance will return equivalent answers.
"""


def generate_vectors(num_vectors: int, num_dim: int,
                     normalize: bool = True) -> np.ndarray:
    """
    Generate random embedding vectors.

    Parameters
    ----------
    num_vectors : int
        Number of vectors to generate.
    num_dim : int
        Dimensionality of the vectors.
    normalize : bool, optional
        Whether to normalize the vectors, by default True.

    Returns
    -------
    np.ndarray
        Randomly generated `num_vectors` vectors with `num_dim` dimensions.
    """
    vectors = np.random.rand(num_vectors, num_dim)
    if normalize:
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors

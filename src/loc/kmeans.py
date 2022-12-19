import numpy as np


def compute_dist(a, b):
    distance = np.sum(a**2, axis=1, keepdims=True) + \
        np.sum(b**2, axis=1, keepdims=True).T - 2 * a@b.T
    return distance

def kmeans(x, k, iterations=1000, eps=1e-3):
    n, d = x.shape
    clusters = x[np.random.choice(n, k, False)]

    for _ in range(iterations):
        # compute distance
        distance = compute_dist(x, clusters)

        # find the nearest cluster center for each sample
        inds = np.argmin(distance, axis=1)

        # update cluster centers
        clusters_ = np.stack([np.mean(x[np.argwhere(inds == j).reshape(-1)], axis=0)
                     for j in range(k)], axis=0)
        diff = np.linalg.norm(clusters-clusters_, axis=1)
        if np.max(diff) < eps:
            break
        clusters = clusters_

    return clusters


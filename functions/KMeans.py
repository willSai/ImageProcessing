from collections import defaultdict
from random import uniform
from math import sqrt, inf

def evaluate_centroids(points):
    """
    Accepts a list of points, each with the same number of dimensions.
    NB. points can have more dimensions than 2

    Returns a new point which is the center of all the points.
    """
    dimensions = len(points[0])

    centroids = []

    for dimension in range(dimensions):
        dim_sum = 0  # dimension sum
        for p in points:
            dim_sum += p[dimension]

        # average of each dimension
        centroids.append(dim_sum / float(len(points)))

    return centroids


def update_centroids(data_set, assignments):
    """
    Accepts a dataset and a list of assignments; the indexes
    of both lists correspond to each other.
    Compute the center for each of the assigned groups.
    Return `k` centers where `k` is the number of unique assignments.
    """
    new_means = defaultdict(list)
    centers = []
    for assignment, point in zip(assignments, data_set):
        new_means[assignment].append(point)

    for c, points in new_means.items():
        centers.append(evaluate_centroids(points))

    return centers


def points_assignment(points, centroids):
    assignments = []
    for point in points:
        min_dist = inf
        min_index = 0
        for i in range(len(centroids)):
            d = distance(point, centroids[i])
            if d < min_dist:
                min_dist = d
                min_index = i
        assignments.append(min_index)
    return assignments


def distance(a, b):
    dimensions = len(a)

    sq_sum = 0
    for dimension in range(dimensions):
        d_sq = (a[dimension] - b[dimension]) ** 2
        sq_sum += d_sq
    return sqrt(sq_sum)


def generate_k_clusters(data_set, k):
    centers = []
    dimensions = len(data_set[0])
    min_max = defaultdict(int)

    for point in data_set:
        for i in range(dimensions):
            val = point[i]
            min_key = 'min_{}'.format(i)
            max_key = 'max_{}'.format(i)
            if min_key not in min_max or val < min_max[min_key]:
                min_max[min_key] = val
            if max_key not in min_max or val > min_max[max_key]:
                min_max[max_key] = val

    for _k in range(k):
        rand_point = []
        for i in range(dimensions):
            min_val = min_max['min_%d' % i]
            max_val = min_max['max_%d' % i]

            rand_point.append(uniform(min_val, max_val))

        centers.append(rand_point)

    return centers


def KMeans(dataset, k):
    k_points = generate_k_clusters(dataset, k)
    assignments = points_assignment(dataset, k_points)
    pre_assignments = None
    centroids = None
    while assignments != pre_assignments:
        new_centroids = update_centroids(dataset, assignments)
        pre_assignments = assignments
        assignments = points_assignment(dataset, new_centroids)
        centroids = new_centroids

    k_clusters = {}
    for i in range(len(centroids)):
        k_clusters[i] = []

    for i in range(len(assignments)):
        point = dataset[i]
        label = assignments[i]
        k_clusters[label].append(point)

    return k_clusters, centroids
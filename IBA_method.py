import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.spatial.distance import euclidean


def indexing(coord1, coord2):
    mean1 = np.mean(coord1)
    mean2 = np.mean(coord2)
    data_length = len(coord1)

    indexing_angles = []
    for i in range(data_length):
        indexing_angle_temp = math.atan2((coord2[i] - mean2), (coord1[i] - mean1)) * 180 / np.pi
        indexing_angles.append(indexing_angle_temp)  # Angle above y axis

    df = pd.DataFrame({
        'coord1': coord1,
        'coord2': coord2,
        'angle': indexing_angles  # indexing_angle (degree)
    })

    sorted_pandas = df.sort_values(by=['angle'], ascending=False)
    sorted = sorted_pandas.to_numpy()

    coord1_idxed = sorted[:, 0]
    coord2_idxed = sorted[:, 1]
    indexing_angles = sorted[:, 2]

    return coord1_idxed, coord2_idxed, np.asarray(indexing_angles)


def main():

    X, Y, Z = np.loadtxt('Section1_000000.txt').T
    sorted_coord1, sorted_coord2, sorted_results_angles = indexing(X, Z)

    ############## AREA CALCULATION ################
    total_area = []
    # # Initialize area
    area = 0.0
    n = int(len(sorted_coord1))
    # # Calculate value of shoelace formula
    j = n - 1
    for i in range(0, n):
        area += (sorted_coord2[j] + sorted_coord2[i]) * (sorted_coord1[j] - sorted_coord1[i])
        j = i  # j is previous vertex to i

    # Return absolute value
    area = abs(area / 2.0)
    total_area.append(area)
    print("Area is [m^2] = ", total_area)
    # print("Count of points", len(sorted_coord1))

    points = []
    for i in range(len(sorted_coord1)):
        points_ = [sorted_coord1[i], sorted_coord2[i]]
        points.append(points_)

    # to calculate perimeter add the first line
    points.append(points[0])
    points = np.array(points)

    hull = ConvexHull(points)

    vertices = hull.vertices.tolist() + [hull.vertices[0]]
    perimeter = np.sum([euclidean(x, y) for x, y in zip(points[vertices], points[vertices][1:])])
    print("Perimeter is [m]: ", perimeter)

    plt.plot(points[:, 0], points[:, 1], 'o')
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
    plt.show()

    # total_distance = []
    # for i in range(len(sorted_coord1)):
    #     distance = math.sqrt(((points[i] - points[i] ** 2) + ((points[i+1] - points[i+1]) ** 2))
    #     total_distance.append(distance)
    #
    # print(total_distance)


if __name__ == '__main__':
    main()

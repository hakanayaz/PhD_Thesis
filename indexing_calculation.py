import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import pandas as pd


def sort_coordinates_y_z(y, z):

    y_mean = np.mean(y)
    z_mean = np.mean(z)
    data_length = len(y)

    indexing_angles = []
    for i in range(data_length):
        indexing_angle_temp = math.atan2((z[i]-z_mean), (y[i]-y_mean))*180/np.pi
        indexing_angles.append(indexing_angle_temp)    # Angle above y axis

    df = pd.DataFrame({
        'y': y,      # y coordinates
        'z': z,      # z coordinates
        'angle': indexing_angles  # indexing_angle (degree)
    })

    Section1_1_sorted_pandas = df.sort_values(by=['angle'], ascending=False)
    Section1_1_sorted = Section1_1_sorted_pandas.to_numpy()

    y_coord_idxed = Section1_1_sorted[:, 0]
    z_coord_idxed = Section1_1_sorted[:, 1]
    indexing_angles = Section1_1_sorted[:, 2]

    return y_coord_idxed, z_coord_idxed, np.asarray(indexing_angles)


def main():
    # Reading the .txt file
    y = np.loadtxt('Rotated_Full_Section1_y.txt').T
    z = np.loadtxt('Rotated_Full_Section1_z.txt').T
    circle_locations = np.loadtxt('circle_locations.txt').T
    y = y.tolist()
    z = z.tolist()

    # for MAKE IT ALL OF THEM
    last_index = 0
    all_circles = []
    for k in circle_locations:
        # k has the last item's location
        # last_index has the first item's location
        k = int(k)
        circle_points = np.arange(last_index, k)
        y_slice = y[last_index:k]
        z_slice = z[last_index:k]
        sorted_results = sort_coordinates_y_z(y_slice, z_slice)
        all_circles.append(sorted_results)
        last_index = k
    print(all_circles[0:1])

    # print("Indexed Y coordinates", y_coord_idxed)
    # print("Indexed Z coordinates", z_coord_idxed)
    # print("Indexed Angles", indexing_angles)
    #
    # fig, ax = plt.subplots()
    # ax.plot(y_coord_idxed, z_coord_idxed, 'o', color='black')
    # ax.plot(y_mean, z_mean, 'o', color='black') # to see where the centroid is
    # plt.show()
    #
    # ax.set(xlabel='Y Coord.', ylabel='Z Coord.',
    #        title='One section how it looks')
    # ax.grid()
    # plt.show()


if __name__ == '__main__':
    main()
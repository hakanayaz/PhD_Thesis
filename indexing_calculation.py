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
    all_circles_y = []
    all_circles_z = []
    all_circles_angles = []
    for k in circle_locations:
        # k has the last item's location
        # last_index has the first item's location
        k = int(k)
        circle_points = np.arange(last_index, k)
        y_slice = y[last_index:k]
        z_slice = z[last_index:k]
        sorted_results_y, sorted_results_z, sorted_results_angles = sort_coordinates_y_z(y_slice, z_slice)
        all_circles_y.append(sorted_results_y)
        all_circles_z.append(sorted_results_z)
        all_circles_angles.append(sorted_results_angles)
        last_index = k


    ############## AREA CALCULATION ################
    # k = 0
    # total_area = []
    # # Initialize area
    # area = 0.0
    # n = int(circle_locations[k])
    # # Calculate value of shoelace formula
    # j = n - 1
    # for i in range(0, n):
    #     area += (all_circles_y[k][j] + all_circles_y[k][i]) * (all_circles_z[k][j] - all_circles_z[k][i])
    #     j = i  # j is previous vertex to i
    #
    # # Return absolute value
    # area = abs(area / 2.0)
    # total_area.append(area)
    #
    # print("Section 1-1 Area is = ", total_area)



    start_area_number = 0
    total_area = []
    for k in range(len(circle_locations)):

        # Initialize area
        area = 0.0
        n = int(circle_locations[k])
        # Calculate value of shoelace formula
        j = n - 1

        for i in range(start_area_number, n):
            area += (all_circles_z[k][j] + all_circles_z[k][i]) * (all_circles_y[k][j] - all_circles_y[k][i])
            j = i  # j is previous vertex to i

        # Return absolute value
        area = abs(area / 2.0)
        total_area.append(area)
        start_area_number = circle_locations[k] + 1

    # Driver program to test above function
    print(total_area)


    # # Calculated data save to the excel file
    # sorted_y = pd.DataFrame(all_circles_y)
    # sorted_z = pd.DataFrame(all_circles_z)
    # sorted_angle = pd.DataFrame(all_circles_angles)
    # # saving the dataframe
    # sorted_y.to_csv('Sorted_y.csv')
    # sorted_z.to_csv('Sorted_z.csv')
    # sorted_angle.to_csv('Sorted_angle.csv')

    # PLOT
    # print("Indexed Y coordinates", y_coord_idxed)
    # print("Indexed Z coordinates", z_coord_idxed)
    # print("Indexed Angles", indexing_angles)
    # fig, ax = plt.subplots()
    # ax.plot(all_circles_y[0], all_circles_z[0], 'o', color='black')
    # #ax.plot(np.mean(all_circles_y[0], np.mean(all_circles_z[0]), 'o', color='black'))  # to see where the centroid is
    # plt.show()
    # ax.set(xlabel='Y Coord.', ylabel='Z Coord.',
    #        title='One section how it looks')
    # ax.grid()
    # plt.show()


if __name__ == '__main__':
    main()
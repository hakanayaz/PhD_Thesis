import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
# import shapely
# from shapely.geometry import Polygon
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.mlab import griddata
from matplotlib import cm


def Rx(theta):
    return np.matrix([[1, 0, 0],
                      [0, math.cos(theta), -math.sin(theta)],
                      [0, math.sin(theta), math.cos(theta)]])


def Rz(theta):
    return np.matrix([[math.cos(theta), -math.sin(theta), 0],
                      [math.sin(theta), math.cos(theta), 0],
                      [0, 0, 1]])


def make_parallel_X_degree(x_data, y_data, z_data):
    x_difference = []
    for t in range(180):
        # Radian converting
        # theta = (0.0174533 * t)
        theta = t
        Rot_Matrix_Z = Rz(math.degrees(theta))
        x_rot = []

        for i in range(len(x_data)):
            point = [[x_data[i]], [y_data[i]], [z_data[i]]]
            # Utilize np.matmul operation for efficiency because matmul using C)
            rotated_mat = np.matmul(Rot_Matrix_Z, point)
            arr_mat = np.squeeze(np.asarray(rotated_mat))
            x_prime = arr_mat[0]
            x_rot.append(x_prime)
        x_diff = max(x_rot) - min(x_rot)
        x_difference.append(x_diff)

    min_deg = x_difference.index(min(x_difference))

    # Store the minimum difference
    min_dif = x_difference[min_deg]

    # print('Parallel to the X axis degree: ', min_deg)
    return min_deg


def rotate_data(x_data, y_data, z_data, theta):
    Rot_Matrix_Z = Rz(math.degrees(theta))
    x_rot = []
    y_rot = []
    z_rot = []
    for i in range(len(x_data)):
        point = [[x_data[i]], [y_data[i]], [z_data[i]]]

        # Utilize np.matmul operation for efficiency because matmul using C
        rotated_mat = np.matmul(Rot_Matrix_Z, point)
        arr_mat = np.squeeze(np.asarray(rotated_mat))
        x_prime = arr_mat[0]
        y_prime = arr_mat[1]
        z_prime = arr_mat[2]
        x_rot.append(x_prime)
        y_rot.append(y_prime)
        z_rot.append(z_prime)
    return x_rot, y_rot, z_rot


def point_circle_location(rotated_data_x):
    # Determine the circle points
    # This is going to be a function and help me to find the circle points
    # After I got the points I will calculate AREA, Perimeter vs...

    diff_xs_loc = []
    point_loc = []
    for k in range(len(rotated_data_x)-1):
        diff_xs1 = np.asarray(rotated_data_x[(k+1)])
        diff_xs0 = np.asarray(rotated_data_x[k])
        x_difference = diff_xs1 - diff_xs0
        x_diff_circles = np.squeeze(np.asarray(x_difference))
        # print(x_circles)
        if abs(x_diff_circles) > 0.55:
            point_loc.append(k+1)  # Storing which point number belongs to that
            diff_xs_loc.append(x_diff_circles)
    # print('Point Circle Locations are :', point_loc)
    # Circle point locations are found !! [point_loc]
    return point_loc


def idxed_coordinates_y_z(rotated_data_y, rotated_data_z, rot_full_mean_y, rot_full_mean_z, circle_locations):

    rot_full_y_mean = []
    rot_full_z_mean = []
    for i in range(len(rotated_data_y)):
        rot_full_y_mean_temp = np.asarray(rotated_data_y[i]-rot_full_mean_y)
        rot_full_y_mean.append(rot_full_y_mean_temp)
        rot_full_z_mean_temp = np.asarray(rotated_data_z[i]-rot_full_mean_z)
        rot_full_z_mean.append(rot_full_z_mean_temp)

    indexing_angles = []
    for i in range(circle_locations[0]):
        indexing_angle_temp = math.atan2(rot_full_z_mean[i], rot_full_y_mean[i])
        indexing_angles.append(indexing_angle_temp)    # Angle above y axis

    cl_start = 1
    cl_end = circle_locations[0]
    df = pd.DataFrame({
        'col1': range(cl_start, cl_end+1),             # old_indexing_number
        'col2': rotated_data_y[cl_start: cl_end+1],      # y coordinates
        'col3': rotated_data_z[cl_start: cl_end+1],      # z coordinates
        'col4': indexing_angles                         # indexing_angle (rad.)
    })

    Section1_1_sorted_pandas = df.sort_values(by=['col4'], ascending=False)
    Section1_1_sorted = Section1_1_sorted_pandas.to_numpy()

    y_coord_idxed = Section1_1_sorted[:, 1]
    z_coord_idxed = Section1_1_sorted[:, 2]
    return y_coord_idxed, z_coord_idxed, indexing_angles


def main():
    # Reading the .txt file
    # .txt file must contain x,y,z values with that order
    X, Y, Z = np.loadtxt('Section1.txt').T

    '''
    Sampling the main data to see while choosing the sampling data maximum 2 verticities must be found 
    because if we increase the points number we cannot find the right angle that represent parallel to the X
    axis.
    '''
    x_sample_4degree = X[0:200]
    y_sample_4degree = Y[0:200]
    z_sample_4degree = Z[0:200]

    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')

    # Plot sampled scatter result
    ax.set_xlabel('X Label', fontsize=10)
    ax.set_ylabel('Y Label', fontsize=10)
    ax.set_zlabel('Z Label', fontsize=10)
    ax.scatter3D(x_sample_4degree, y_sample_4degree, z_sample_4degree, c=z_sample_4degree, cmap=plt.cm.jet)

    # To be able to calculate angle that needs to rotate and
    # Make it parallel to the X axis.
    min_degree = make_parallel_X_degree(x_sample_4degree, y_sample_4degree, z_sample_4degree)
    rot_data_sample_4degree = rotate_data(x_sample_4degree, y_sample_4degree, z_sample_4degree, min_degree)
    rotated_data_x_sample = rot_data_sample_4degree[0]
    rotated_data_y_sample = rot_data_sample_4degree[1]
    rotated_data_z_sample = rot_data_sample_4degree[2]

    # Subplot helped me to graph together 122 - 121
    ax2 = fig.add_subplot(122, projection='3d')

    # Plot rotated points and sampled scatter result
    ax2.set_xlabel('X Label', fontsize=10)
    ax2.set_ylabel('Y Label', fontsize=10)
    ax2.set_zlabel('Z Label', fontsize=10)
    ax2.scatter3D(rotated_data_x_sample, rotated_data_y_sample, rotated_data_z_sample,
                  c=rotated_data_z_sample, cmap=plt.cm.jet)

    '''
    Plotting the data for getting the angle for paralleling the plot to the X axis.
    Now we have angle can parallel to the X axis.
    '''
    # plt.show()

    ''' FULL DATA Calculation !!
    After finding the min angle apply to the full data
    '''
    rot_data_full = rotate_data(X, Y, Z, min_degree)
    rotated_data_x = rot_data_full[0]
    rotated_data_y = rot_data_full[1]
    rotated_data_z = rot_data_full[2]

    # To save all the data from Section-1 after rotated.
    # np.savetxt("Rotated_Full_Section1_x.txt", rotated_data_x)
    # np.savetxt("Rotated_Full_Section1_y.txt", rotated_data_y)
    # np.savetxt("Rotated_Full_Section1_z.txt", rotated_data_z)


    ''' for Full Data Plot  '''
    # Full_Plot = plt.figure()
    # ax_full = Axes3D(Full_Plot)
    # ax_full.scatter3D(rotated_data_x, rotated_data_y, rotated_data_z, c=rotated_data_z, cmap=plt.cm.jet)
    # plt.show()

    # Determine the circle points
    circle_locations = point_circle_location(rotated_data_x)
    ########### To be able to see which numbers circle locations have #############
    # print(circle_locations)


    ########################  INDEXING   ##########################

    rot_full_mean_y = np.mean(rotated_data_y)
    # print("Mean of the y values are:", rot_full_mean_y)
    rot_full_mean_z = np.mean(rotated_data_z)
    # print("Mean of the z values are:", rot_full_mean_z)

    # for MAKE IT ALL OF THEM
    # last_index=1
    # cir_idx=0
    #
    # for k in range(last_index, circle_locations[cir_idx]):
    #     ## whatever you want to do
    #     last_index = circle_locations[cir_idx]+1
    #     cir_idx = cir_idx+1

    [y_coord_idxed, z_coord_idxed, indexing_angles] = idxed_coordinates_y_z(rotated_data_y,rotated_data_z,rot_full_mean_y,rot_full_mean_z,circle_locations)

    print("Indexing Angles are: ", indexing_angles)


    # Area Calculations are true and it req to clean up the data
    total_area = []
    # Initialize area
    area = 0.0
    n = circle_locations[0]
    # Calculate value of shoelace formula
    j = n - 1
    for i in range(0, n):
        area += (y_coord_idxed[j] + y_coord_idxed[i]) * (z_coord_idxed[j] - z_coord_idxed[i])
        j = i  # j is previous vertex to i

    # Return absolute value
    area = abs(area / 2.0)
    total_area.append(area)

    print("Section 1-1 Area is = ", total_area)





    ############### AREA CALCULATION ################
    # start_area_number = 0
    # total_area = []
    # for k in range(len(circle_locations)):
    #
    #     # Initialize area
    #     area = 0.0
    #     n = circle_locations[k]-5
    #     # Calculate value of shoelace formula
    #     j = n - 1
    #     for i in range(start_area_number, n):
    #         area += (rotated_data_z[j] + rotated_data_z[i]) * (rotated_data_y[j] - rotated_data_y[i])
    #         j = i  # j is previous vertex to i
    #
    #     # Return absolute value
    #     area = abs(area / 2.0)
    #     total_area.append(area)
    #     start_area_number = circle_locations[k] + 1
    #
    # # Driver program to test above function
    # print(total_area)





    '''
    After find the points that represents area use Gauss's area formula to
    find the area of the section. Find surface Roughness!!
    Surface roughness is a unitless peace of the equation that's why you can calculate easily
    '''


if __name__ == '__main__':
    main()

import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.mlab import griddata
from matplotlib import cm


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

    ''' for Full Data Plot  '''
    # Full_Plot = plt.figure()
    # ax_full = Axes3D(Full_Plot)
    # ax_full.scatter3D(rotated_data_x, rotated_data_y, rotated_data_z, c=rotated_data_z, cmap=plt.cm.jet)
    # plt.show()

    # Determine the circle points
    circle_locations = point_circle_location(rotated_data_x)
    ########### To be able to see which numbers circle locations have #############
    # print(circle_locations)

    '''
    After find the points that represents area use Gauss's area formula to
    find the area of the section. Find surface Roughness!!
    Surface roughness is a unitless peace of the equation that's why you can calculate easily
    '''

    start = 0
    tot_heights = []
    for i in range(len(circle_locations)):
        # storage_val = circle_locations[i]
        section_heights = []
        for j in range(start, circle_locations[i]+1):
            section_rotated_height = rotated_data_z[j]
            section_heights.append(section_rotated_height)
        section_height_average = np.average(section_heights)
        tot_heights.append(section_height_average)
        start = circle_locations[i]+1
    # print(tot_heights)
    mean_total_height =np.mean(tot_heights)
    print("Mean Total height is: ", mean_total_height)


    ki = []
    for i in range(len(circle_locations)):
        ki_cal = np.sqrt((1/len(circle_locations) * np.square(tot_heights[i] - mean_total_height)))
    ki.append(ki_cal)
    print("ki is equal to: ", ki)



if __name__ == '__main__':
    main()

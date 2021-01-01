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

def find_miniumum_degree(x_data, y_data, z_data):
    x_difference = []
    for t in range(180):
        theta = -(t+1)
        Rot_Matrix_Z = Rz(theta)
        x_rot = []

        for i in range(len(x_data)):
            point = [[x_data[i]], [y_data[i]], [z_data[i]]]
            # Utilize np.matmul operation for efficiency because matmul using C
            rotated_mat = np.matmul(Rot_Matrix_Z, point)
            arr_mat = np.squeeze(np.asarray(rotated_mat))
            x_prime = arr_mat[0]
            x_rot.append(x_prime)

        x_diff = max(x_rot) - min(x_rot)
        x_difference.append(x_diff)
    
    min_deg = x_difference.index(min(x_difference))
    # store the minimum difference
    min_dif = x_difference[min_deg]

    return min_deg

def rotate_data(x_data, y_data, z_data, theta):
    Rot_Matrix_Z = Rz(theta)
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

def main():
    # Reading the .txt file
    # .txt file must contain x,y,z values with that order
    X, Y, Z = np.loadtxt('Section1.txt').T

    # Sampling the main data to see
    x = X[0:100]
    y = Y[0:100]
    z = Z[0:100]

    ############  PLOATING  #############
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')

    # Plot sampled scatter result
    ax.set_xlabel('X Label', fontsize=10)
    ax.set_ylabel('Y Label', fontsize=10)
    ax.set_zlabel('Z Label', fontsize=10)
    ax.scatter3D(x, y, z, c=z, cmap=plt.cm.jet)

    
    min_degree = find_miniumum_degree(x,y,z)
    rot_data = rotate_data(x,y,z,min_degree)
    rotated_data_x = rot_data[0]
    rotated_data_y = rot_data[1]
    rotated_data_z = rot_data[2]

    # Subplot helped me to graph together 122 - 121
    ax2 = fig.add_subplot(122, projection='3d')

    # Plot rotated points and sampled scatter result
    ax2.set_xlabel('X Label', fontsize=10)
    ax2.set_ylabel('Y Label', fontsize=10)
    ax2.set_zlabel('Z Label', fontsize=10)
    ax2.scatter3D(rotated_data_x, rotated_data_y, rotated_data_z, c=rotated_data_z, cmap=plt.cm.jet)

    # [TOSEE] Plotting the data
    plt.show()


if __name__ == '__main__':
    main()

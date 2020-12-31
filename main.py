import numpy as np
import matplotlib.pyplot as plt
import math as m
from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.mlab import griddata
from matplotlib import cm


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
    # fig = plt.figure()
    # ax = Axes3D(fig)

    # Plot sampled scatter result
    ax.set_xlabel('X Label', fontsize=10)
    ax.set_ylabel('Y Label', fontsize=10)
    ax.set_zlabel('Z Label', fontsize=10)
    ax.scatter3D(x, y, z, c=z, cmap=plt.cm.jet)

    # [TOSEE] Plotting the data
    # plt.show()

    ############  ROTATING  #############
    # Rotating the Point Cloud to be able to calculate the area

    # Finding the [theta]
    theta = m.pi / 6 * -1

    # print(theta)

    # Z Axis rotation definition
    def Rz(theta):
        return np.matrix([[m.cos(theta), -m.sin(theta), 0],
                          [m.sin(theta), m.cos(theta), 0],
                          [0, 0, 1]])

    Rot_Matrix_Z = Rz(theta)
    Rot_Points = []

    # print(Rot_Matrix_Z)
    x_rot = []
    y_rot = []
    z_rot = []
    for i in range(len(x)):
        point = [[x[i]], [y[i]], [z[i]]]

        # Utilize numpy operation for efficency
        rotated_mat = np.matmul(Rot_Matrix_Z, point)

        arr_mat = np.squeeze(np.asarray(rotated_mat))
        x_prime = arr_mat[0]
        y_prime = arr_mat[1]
        z_prime = arr_mat[2]
        x_rot.append(x_prime)
        y_rot.append(y_prime)
        z_rot.append(z_prime)

    ax2 = fig.add_subplot(122, projection='3d')
    # fig = plt.figure()
    # ax = Axes3D(fig)

    # Plot sampled scatter result
    ax2.set_xlabel('X Label', fontsize=10)
    ax2.set_ylabel('Y Label', fontsize=10)
    ax2.set_zlabel('Z Label', fontsize=10)
    ax2.scatter3D(x_rot, y_rot, z_rot, c=z_rot, cmap=plt.cm.jet)

    # [TOSEE] Plotting the data
    plt.show()


if __name__ == '__main__':
    main()

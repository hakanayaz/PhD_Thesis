import numpy as np
import matplotlib.pyplot as plt
import math as m
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.mlab import griddata
from matplotlib import cm


def main():

    # Reading the .txt file
    # .txt file must contain x,y,z values with that order
    X, Y, Z = np.loadtxt('Section1.txt').T

    # Sampling the main data to see
    x = X[0:72]
    y = Y[0:72]
    z = Z[0:72]

    ############  PLOATING  #############
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
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
    theta = m.pi/6 * -1
    # print(theta)

    # Z Axis rotation definition
    def Rz(theta):
        return np.matrix([[m.cos(theta), -m.sin(theta), 0],
                          [m.sin(theta), m.cos(theta), 0],
                          [0, 0, 1]])

    Rot_Matrix_Z = Rz(theta)
    Rot_Points = []
    # print(Rot_Matrix_Z)
    for i in range(len(x)):
        Point = np.transpose([X[i], Y[i], Z[i]])
        Rot_Point = (Point * Rot_Matrix_Z)
        Rot_Points.append(Rot_Point)  # Store value retrieved from a for Loop

    # Find efficient way to store values and plot calculate :)



if __name__ == '__main__':
    main()

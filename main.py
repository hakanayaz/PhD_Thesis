import numpy
import matplotlib.pyplot
import math
from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.mlab import griddata
from matplotlib import cm


def main():
    # Reading the .txt file
    # .txt file must contain x,y,z values with that order
    X, Y, Z = numpy.loadtxt('Section1.txt').T

    # Sampling the main data to see
    x = X[0:100]
    y = Y[0:100]
    z = Z[0:100]

    ############  PLOATING  #############
    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(121, projection='3d')

    # Plot sampled scatter result
    ax.set_xlabel('X Label', fontsize=10)
    ax.set_ylabel('Y Label', fontsize=10)
    ax.set_zlabel('Z Label', fontsize=10)
    ax.scatter3D(x, y, z, c=z, cmap=matplotlib.pyplot.cm.jet)

    # [TOSEE] Plotting the data
    # plt.show()

    ############  ROTATING  #############
    # Rotating the Point Cloud to be able to calculate the area

    # Finding the [theta]
    # theta = math.pi / 6 * -1

    # Z Axis rotation definition
    def Rz(theta):
        return numpy.matrix([[math.cos(theta), -math.sin(theta), 0],
                          [math.sin(theta), math.cos(theta), 0],
                          [0, 0, 1]])

    ##################################################################################

    # Rot_Matrix_Z = Rz(theta)

    # x_rot = []
    # y_rot = []
    # z_rot = []
    # for i in range(len(x)):
        # point = [[x[i]], [y[i]], [z[i]]]

        # # Utilize numpy.matmul operation for efficiency because matmul using C
        # rotated_mat = numpy.matmul(Rot_Matrix_Z, point)
        # arr_mat = numpy.squeeze(numpy.asarray(rotated_mat))
        # x_prime = arr_mat[0]
        # y_prime = arr_mat[1]
        # z_prime = arr_mat[2]
        # x_rot.append(x_prime)
        # y_rot.append(y_prime)
        # z_rot.append(z_prime)

    # # print("Original points X difference", max(x) - min(x))
    # # print("Rotating points X  difference", max(x_rot) - min(x_rot))

    # # Subplot helped me to graph together 122 - 121
    # ax2 = fig.add_subplot(122, projection='3d')

    # # Plot rotated points and sampled scatter result
    # ax2.set_xlabel('X Label', fontsize=10)
    # ax2.set_ylabel('Y Label', fontsize=10)
    # ax2.set_zlabel('Z Label', fontsize=10)
    # ax2.scatter3D(x_rot, y_rot, z_rot, c=z_rot, cmap=matplotlib.pyplot.cm.jet)

    # # [TOSEE] Plotting the data
    # matplotlib.pyplot.show()

    ##################################################################################

    # theta = math.pi / 6 * -1
    for t in range(180):

        # theta = (math.pi / (t+1)) * -1
        theta = (t+1)
        Rot_Matrix_Z = Rz(theta)

        x_rot = []
        y_rot = []
        z_rot = []
        x_difference = []
        x_diff

        for i in range(len(x)):
            point = [[x[i]], [y[i]], [z[i]]]

            # Utilize numpy.matmul operation for efficiency because matmul using C
            rotated_mat = numpy.matmul(Rot_Matrix_Z, point)
            arr_mat = numpy.squeeze(numpy.asarray(rotated_mat))
            x_prime = arr_mat[0]
            y_prime = arr_mat[1]
            z_prime = arr_mat[2]
            x_rot.append(x_prime)
            y_rot.append(y_prime)
            z_rot.append(z_prime)

            x_diff = max(x_rot) - min(x_rot)
        x_difference.append(x_diff)
        print(len(x_difference))

    # print("Original points X difference", max(x) - min(x))
    # print("Rotating points X  difference", max(x_rot) - min(x_rot))

    # Subplot helped me to graph together 122 - 121
    ax2 = fig.add_subplot(122, projection='3d')

    # Plot rotated points and sampled scatter result
    ax2.set_xlabel('X Label', fontsize=10)
    ax2.set_ylabel('Y Label', fontsize=10)
    ax2.set_zlabel('Z Label', fontsize=10)
    ax2.scatter3D(x_rot, y_rot, z_rot, c=z_rot, cmap=matplotlib.pyplot.cm.jet)

    # [TOSEE] Plotting the data
    # matplotlib.pyplot.show()


if __name__ == '__main__':
    main()

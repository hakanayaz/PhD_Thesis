import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.mlab import griddata
from matplotlib import cm


def main():

    # Reading the .txt file
    # .txt file must contain x,y,z values with that order
    X, Y, Z = np.loadtxt('Section1.txt').T

    # Sampling the main data
    x = X[0:1000]
    y = Y[0:1000]
    z = Z[0:1000]

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    fig = plt.figure()
    ax = Axes3D(fig)

    # Plot a basic wireframe result of the first 100 points
    # Axes3D.plot_wireframe(a, b, c)
    ax.scatter3D(x, y, z, c=z, cmap=plt.cm.jet)

    # ax.plot_wireframe(x, y, z, rstride=10, cstride=10)

    # Ploting the data
    plt.show()


if __name__ == '__main__':
    main()

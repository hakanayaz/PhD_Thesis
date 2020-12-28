import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def main():

    # Reading the .txt file
    # .txt file must contain x,y,z values with that order
    x, y, z = np.loadtxt('Section1.txt').T

    # Working...
    #print("First 10 Z values are = \n", z[0:10])

    a = x[0:100]
    b = y[0:100]
    c = z[0:100]

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # Plot a basic wireframe result of the first 100 points
    Axes3D.plot_wireframe(a, b, c)
    #Axes3D.plot_wireframe(a, b, c, rstride=10, cstride=10)
    plt.show()


if __name__ == '__main__':
    main()

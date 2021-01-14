import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import pyplot
import spline as spline
from mpl_toolkits.mplot3d import Axes3D
#from matplotlib.mlab import griddata
from matplotlib import cm
import scipy as sp
import scipy.interpolate.ndgriddata
import plotly.graph_objects as go


def main():
    # Reading the .txt file
    # .txt file must contain x,y,z values with that order
    X_main, Y_main, Z_main = np.loadtxt('Section1.txt').T

    x_sample_4degree = X_main[0:5000]
    y_sample_4degree = Y_main[0:5000]
    z_sample_4degree = Z_main[0:5000]

    # xi = np.linspace(min(x_sample_4degree), max(x_sample_4degree))
    # yi = np.linspace(min(y_sample_4degree), max(y_sample_4degree))
    # X, Y = np.meshgrid(xi, yi)
    # interpolation
    # z = scipy.interpolate.griddata(x_sample_4degree, y_sample_4degree, z_sample_4degree, xi, yi, interp='linear')

########################
    # z = z_sample_4degree
    # sh_0, sh_1 = z.shape
    # x, y = np.linspace(0, 1, sh_0), np.linspace(0, 1, sh_1)
    # fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
    # fig.update_layout(title='Surface Roughness', autosize=False,
    #                   width=500, height=500,
    #                   margin=dict(l=65, r=50, b=65, t=90))
    # fig.show()

#######################

    fig = plt.figure()
    # ax = fig.add_subplot(121, projection='3d')
    # # ax =Axes3D(fig)
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=1, antialiased=True)
    # #plt.show()
    #
    ax2 = fig.add_subplot(111, projection='3d')
    # Plot sampled scatter result
    ax2.set_xlabel('X Label', fontsize=10)
    ax2.set_ylabel('Y Label', fontsize=10)
    ax2.set_zlabel('Z Label', fontsize=10)
    ax2.scatter3D(x_sample_4degree, y_sample_4degree, z_sample_4degree, c=z_sample_4degree, cmap=plt.cm.jet)

    plt.show()


if __name__ == '__main__':
    main()

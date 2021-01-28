import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
# import matplotlib as mpl
# from matplotlib import pyplot
# import spline as spline
# from mpl_toolkits.mplot3d import Axes3D
# #from matplotlib.mlab import griddata
# from matplotlib import cm
# import scipy as sp
# import scipy.interpolate.ndgriddata
# import plotly.graph_objects as go


def main():
    # Reading the .txt file
    # .txt file must contain x,y,z values with that order
    # X_main, Y_main, Z_main = np.loadtxt('Section1.txt').T
    #
    # x_sample_4degree = X_main[:72]
    # y_sample_4degree = Y_main[:72]
    # z_sample_4degree = Z_main[:72]

    # create paths and load data
    point_cloud = np.loadtxt('Section1.txt', skiprows=1)
    # Format to open3d usable objects
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:6] / 255)
    pcd.normals = o3d.utility.Vector3dVector(point_cloud[:, 6:9])

    # radius determination
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 3 * avg_dist

    # computing the mesh
    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(
        [radius, radius * 2]))
    # decimating the mesh
    dec_mesh = bpa_mesh.simplify_quadric_decimation(100000)





    # print("Load a ply point cloud, print it, and render it")
    # pcd = o3d.io.read_point_cloud("Section1.ply")
    # print(pcd)
    # print(np.asarray(pcd.points))
    # # o3d.visualization.draw_geometries([pcd])
    #                                 # zoom=0.3412,
    #                                 # front=[0.4257, -0.2125, -0.8795],
    #                                 # lookat=[2.6172, 2.0475, 1.532],
    #                                 # up=[-0.0694, -0.9768, 0.2024])
    # alpha = 2
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    # mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)


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

    # fig = plt.figure()
    # # ax = fig.add_subplot(121, projection='3d')
    # # # ax =Axes3D(fig)
    # # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=1, antialiased=True)
    # # #plt.show()
    # #
    # ax2 = fig.add_subplot(111, projection='3d')
    # # Plot sampled scatter result
    # ax2.set_xlabel('X Label', fontsize=10)
    # ax2.set_ylabel('Y Label', fontsize=10)
    # ax2.set_zlabel('Z Label', fontsize=10)
    # ax2.scatter3D(x_sample_4degree, y_sample_4degree, z_sample_4degree, c=z_sample_4degree, cmap=plt.cm.jet)
    #
    plt.show()

if __name__ == '__main__':
    main()

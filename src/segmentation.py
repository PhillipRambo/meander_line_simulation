import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

def generate_meander_antenna(lambda_g, d, s, L, w, N=4, num_points=200, plot=True):

    total_height = (N - 1) * d + N * w
    line_x = []
    line_y = []

    # Start at bottom of last vertical segment
    x_last = -s/2 - w/2
    y_last_bottom = -total_height/2 - d/2
    y_last_top = -total_height/2
    line_x.append(x_last)
    line_y.append(y_last_bottom)
    line_x.append(x_last)
    line_y.append(y_last_top)

    # Build line along meander
    for i in range(N):
        y_center = i * (d + w) - total_height/2 + w/2
        # Horizontal segment
        if i % 2 == 0:
            line_x.append(-s/2)
            line_y.append(y_center)
            line_x.append(s/2)
            line_y.append(y_center)
        else:
            line_x.append(s/2)
            line_y.append(y_center)
            line_x.append(-s/2)
            line_y.append(y_center)
        # Vertical connector (except last)
        if i < N - 1:
            x_vert = (s/2 + w/2) if i % 2 == 0 else (-s/2 - w/2)
            y_bottom_vert = y_center + w/2
            y_top_vert = y_bottom_vert + d
            line_x.append(x_vert)
            line_y.append(y_bottom_vert)
            line_x.append(x_vert)
            line_y.append(y_top_vert)

    # Discretize line
    distances = [0]
    for i in range(1, len(line_x)):
        dx = line_x[i] - line_x[i-1]
        dy = line_y[i] - line_y[i-1]
        distances.append(distances[-1] + np.sqrt(dx**2 + dy**2))
    distances = np.array(distances)
    distance_uniform = np.linspace(0, distances[-1], num_points)
    x_uniform = np.interp(distance_uniform, distances, line_x)
    y_uniform = np.interp(distance_uniform, distances, line_y)
    discrete_points = np.vstack((x_uniform, y_uniform)).T

    if plot:
        fig, ax = plt.subplots()
        # Draw antenna segments
        for i in range(N):
            y_bottom = i * (d + w) - total_height/2
            x_left = -s/2
            x_right = s/2

            # Horizontal segment
            rect_horizontal = plt.Rectangle((x_left, y_bottom), s, w, color="skyblue", ec="k")
            ax.add_patch(rect_horizontal)

            # Vertical connector
            if i < N - 1:
                x_vert = x_right if i % 2 == 0 else x_left - w
                y_vert = y_bottom + w
                rect_vertical = plt.Rectangle((x_vert, y_vert), w, d, color="skyblue", ec="k")
                ax.add_patch(rect_vertical)

            # Tapers
            if i % 2 == 0:
                taper_right = Polygon([(x_right, y_bottom), (x_right + w, y_bottom + w), (x_right, y_bottom + w)], closed=True, color="skyblue", ec="k")
                taper_left = Polygon([(x_left, y_bottom), (x_left - w, y_bottom), (x_left, y_bottom + w)], closed=True, color="skyblue", ec="k")
            else:
                taper_right = Polygon([(x_right, y_bottom), (x_right + w, y_bottom), (x_right, y_bottom + w)], closed=True, color="skyblue", ec="k")
                taper_left = Polygon([(x_left, y_bottom), (x_left - w, y_bottom + w), (x_left, y_bottom + w)], closed=True, color="skyblue", ec="k")

            ax.add_patch(taper_right)
            ax.add_patch(taper_left)

        # Last vertical segment
        rect_vertical_low = plt.Rectangle((-s/2, -total_height/2), -w, -d/2, color="skyblue", ec="k")
        ax.add_patch(rect_vertical_low)

        # Plot centerline and points
        ax.plot(line_x, line_y, color="red", linewidth=1.5, zorder=5)
        ax.scatter(discrete_points[:, 0], discrete_points[:, 1], color='black', s=5, zorder=6)

        ax.set_aspect("equal")
        ax.set_xlim(-s, s)
        ax.set_ylim((-L/2) * 1.5, L/2)
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title("Meander Line Antenna - Segmented Structure")
        plt.show()

    return discrete_points


def generate_3d_structure(points_2D, wire_height, plot=True):
    segments_3D = []
    array_2D = np.asarray(points_2D)

    for i in range(len(array_2D)):
        x, y = array_2D[i]
        z = wire_height
        segments_3D.append([x, y, z])
    segments_3D = np.array(segments_3D)    
    
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(segments_3D[:, 0], segments_3D[:, 1], segments_3D[:, 2])
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.show(block=False)

        return np.array(segments_3D)
    else:
        return np.array(segments_3D)






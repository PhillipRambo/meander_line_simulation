
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from segmentation import generate_meander_antenna, generate_3d_structure
import numpy as np

def observer_relation_locally(observation_point, source_points, plot=True):
    s_array = []
    s_hat_array = []
    R_array = []
    h_list = []
    h_hat_array = []
    observation_point = np.asarray(observation_point).reshape(-1)
    
    center_points = (source_points[:-1] + source_points[1:]) / 2  # centering the observation point
    for i in range(len(center_points)):
        single_point_center = center_points[i]
        single_point_segment = source_points[i]
        # Vector to observer
        s = observation_point - single_point_center
        R = np.linalg.norm(s)
        s_hat = s / R

        # Dipole length (distance to next dot, skip last)
        if i < len(source_points) - 1:
            h = source_points[i+1] - single_point_segment
            h_abs = np.linalg.norm(h)
            h_hat = h/h_abs
            h_list.append(h_abs)
            h_hat_array.append(h_hat)


        # Store vectors and distance
        s_array.append(s)
        s_hat_array.append(s_hat)
        R_array.append(R)
    if plot:
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        points_3d_arr = np.array(source_points)
        ax.plot(points_3d_arr[:,0], points_3d_arr[:,1], points_3d_arr[:,2], 'bo-', label='Antenna dots')
        ax.scatter(observation_point[0], observation_point[1], observation_point[2], c='r', s=20, label='Observer')

        # Plot vectors from each dot to observer
        for s_vec, point in zip(s_array, center_points):
            ax.plot([point[0], point[0]+s_vec[0]],
                    [point[1], point[1]+s_vec[1]],
                    [point[2], point[2]+s_vec[2]], 'g-', alpha=0.5)
            
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Antenna points and vectors to observer')
        ax.legend()
        ax.grid(True)
        plt.show()   
        return s_array, s_hat_array, R_array, h_list, h_hat_array, center_points
    else:
        return s_array, s_hat_array, R_array, h_list, h_hat_array, center_points



def plot_vectors(points_3d, h_hat_array, lengths, N=1):
    points_3d = np.array(points_3d)
    h_hat_array = np.array(h_hat_array)
    lengths = np.array(lengths)

    origins = points_3d[:N]
    vectors = h_hat_array[:N] * lengths[:N, None]

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection="3d")

    # plot N vectors
    ax.quiver(
        origins[:,0], origins[:,1], origins[:,2],
        vectors[:,0], vectors[:,1], vectors[:,2],
        color="blue", normalize=False, label="Vectors"
    )
    ax.scatter(
        points_3d[:,0], points_3d[:,1], points_3d[:,2],
        c="red", s=40, label="Points"
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"First {N} Vectors at Points")
    ax.legend()

    # auto scale
    all_data = np.vstack([points_3d, origins + vectors])
    max_val = np.max(np.abs(all_data)) 
    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    ax.set_zlim([-max_val, max_val])

    plt.show()



def observer_relation_globally(s_hat, h_hat):
    s_hat = np.asarray(s_hat)
    h_hat = np.asarray(h_hat)

    phi_hat = np.cross(h_hat, s_hat) / np.linalg.norm(np.cross(h_hat, s_hat))
    r_hat = s_hat

    theta_hat = np.cross(phi_hat, s_hat)/(np.linalg.norm(np.cross(phi_hat, s_hat)))

    sin_theta = np.linalg.norm(np.cross(s_hat, h_hat), axis=1)
    cos_theta = np.einsum('ij,ij->i', s_hat, h_hat)

    return phi_hat, r_hat, theta_hat, sin_theta, cos_theta



import numpy as np
import matplotlib.pyplot as plt

def current_distribution(h_list, wavelength, amplitude=1.0, plot=False):
    # Segment boundaries
    boundaries = np.cumsum(h_list)
    positions = np.insert(boundaries, 0, 0)
    center_points = (positions[:-1] + positions[1:]) / 2  
    total_length = positions[-1]

    # Current distribution: sinusoidal wave with zero at the right end
    currents = amplitude * np.sin(2 * np.pi * (center_points - total_length) / wavelength)

    if plot:
        x_vals = np.linspace(0, total_length, 1000)
        y_vals = amplitude * np.sin(2 * np.pi * (x_vals - total_length) / wavelength)

        plt.hlines(0, 0, total_length, colors="black", linewidth=2)

        for x in positions:
            plt.vlines(x, -0.2, 0.2, colors="red")

        plt.plot(x_vals, y_vals, color="gray", linestyle="--", label="Sinusoidal current")
        plt.scatter(center_points, currents, color="blue", zorder=5, label="Currents at centers")

        plt.title("Current distribution along antenna")
        plt.xlabel("Length")
        plt.ylabel("Current amplitude")
        plt.legend()
        plt.show()

    return center_points, currents


import numpy as np
import matplotlib.pyplot as plt

def segment_integrals(h_array, wavelength, amplitude=1.0, L=None, plot=False):
    """
    h_array:    array of segment lengths (length N)
    wavelength: scalar
    amplitude:  amplitude A of I(x) = A * sin(k*(x-L))
    L:          right-end coordinate. If None, inferred as sum(h_array)
    plot:       if True, show the continuous current and segment integrals
    Returns: I_seg (integrated current over each segment, units A·m)
    """
    h = np.asarray(h_array)
    left_edges = np.insert(np.cumsum(h)[:-1], 0, 0)  # left edge of each segment
    right_edges = left_edges + h

    if L is None:
        L = np.sum(h)

    k = 2 * np.pi / wavelength

    # analytic segment integrals
    I_seg = -amplitude / k * (np.cos(k * (right_edges - L)) - np.cos(k * (left_edges - L)))

    # handle extremely large wavelength (k -> 0)
    if np.isclose(k, 0.0):
        I_seg = amplitude * 0.5 * (right_edges**2 - left_edges**2 - 2*L*(right_edges - left_edges))

    if plot:
        x_vals = np.linspace(0, L, 1000)
        I_vals = amplitude * np.sin(k * (x_vals - L))

        centers = (left_edges + right_edges) / 2
        I_avg = I_seg / h  # average current per segment

        plt.figure(figsize=(8,4))
        plt.plot(x_vals, I_vals, "k--", label="Continuous I(x)")
        plt.hlines(0, 0, L, colors="gray", linewidth=1)

        for i in range(len(h)):
            plt.bar(centers[i], I_avg[i], width=h[i], alpha=0.4, align="center",
                    label="Segment avg" if i==0 else "")

        plt.scatter(centers, I_avg, color="red", zorder=5, label="Avg current per segment")
        plt.xlabel("Position along wire")
        plt.ylabel("Current (A)")
        plt.title("Continuous vs. segment-integrated current distribution")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return I_seg

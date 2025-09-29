import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from segmentation import generate_meander_antenna, generate_3d_structure
from local_evaluation import plot_vectors, observer_relation_globally, observer_relation_locally, current_distribution
from evaluate_fields import compute_fields_for_points, create_uniform_sphere, calculate_directivity
import numpy as np


def create_uniform_sphere(obs_distance, num_points, plot=True):
    indices = np.arange(0, num_points, dtype=float) + 0.5
    phi = 2 * np.pi * indices / ((1 + np.sqrt(5)) / 2)
    theta = np.arccos(1 - 2*indices/num_points)

    X = obs_distance * np.sin(theta) * np.cos(phi)
    Y = obs_distance * np.sin(theta) * np.sin(phi)
    Z = obs_distance * np.cos(theta)
    if plot:
        ### Print spherical surface points
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(X, Y, Z)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.show()
        return np.stack([X, Y, Z], axis=1)
    else:
        return np.stack([X, Y, Z], axis=1)

def plot_magnitude_radiation(observation_points, field_array, dB=True, label=''):
    E_mag_linear = np.linalg.norm(field_array, axis=1)
    obs_norm = observation_points / np.linalg.norm(observation_points, axis=1)[:, None]

    # Scale points radially by linear E-field magnitude
    scaled_points = obs_norm * E_mag_linear[:, None]

    # Compute magnitude in dB for coloring if requested
    if dB:
        # Convert linear magnitude to dB relative to 1 V/m
        E_color = 20 * np.log10(E_mag_linear + 1e-20)  # small number to avoid log(0)
    else:
        E_color = E_mag_linear

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    p = ax.scatter(
        scaled_points[:, 0], scaled_points[:, 1], scaled_points[:, 2],
        c=E_color, cmap='turbo'
    )

    # Colorbar and titles
    if dB:
        fig.colorbar(p, label=f'Radiated {label}-Field [dB V/m]')
        ax.set_title(f'Radiated {label}-Field in dB')
    else:
        fig.colorbar(p, label=f'Radiated {label}-Field [V/m]')
        ax.set_title(f'Radiated {label}-Field')

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')

    plt.show()



lambda_g = 179e-3
d = 0.16 * lambda_g
s = 0.42 * lambda_g
L = 0.7 * lambda_g
w = 0.05 * lambda_g
wire_height = 35e-6
num_points = 100
Freq_of_interest = 1060e6 #1060MHz
wavelength = 3.0 * 10**8 / Freq_of_interest
eta = 377 #Ohms.. intrinsic impedance, not sure of this
k = 2 * np.pi / lambda_g

obs_point = np.array([[1, 1, 1]])
obs_points = create_uniform_sphere(obs_distance=wavelength*10, num_points=3000, plot=False)
points = generate_meander_antenna(lambda_g, d, s, L, w, N=4, num_points=num_points, plot=False) 
points_3d = generate_3d_structure(points_2D=points, wire_height=wire_height, plot=False) 
s_array, s_hat_array, r_array, h_list, h_hat_array, center_points = observer_relation_locally(obs_point, points_3d, plot=False)
_, I0 = current_distribution(h_list, wavelength, amplitude=1.0, plot=False)


# Compute fields
E_total_array, H_total_array, E_total_array_complex, H_total_array_complex = compute_fields_for_points(obs_points, points_3d, eta, k, I0)


#plot_magnitude_radiation(obs_points, E_total_array, dB=True, label='E')
#plot_magnitude_radiation(obs_points, H_total_array, dB=True, label='H')

D0 = calculate_directivity(E_total_array_complex, H_total_array_complex, observationpoints=obs_points)
print(D0)
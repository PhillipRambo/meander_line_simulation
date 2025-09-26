import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from segmentation import generate_meander_antenna, generate_3d_structure
from local_evaluation import plot_vectors, observer_relation_globally, observer_relation_locally
from evaluate_fields import compute_fields_for_points
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



lambda_g = 179e-3

d = 0.16 * lambda_g

s = 0.42 * lambda_g

L = 0.7 * lambda_g

w = 0.05 * lambda_g

wire_height = 0.1

num_points = 10

Freq_of_interest = 1060e6 #1060MHz

wavelength = 3.0 * 10**8 / Freq_of_interest

k = 2 * np.pi / wavelength
eta = 377 #Ohms.. intrinsic impedance, not sure of this
I0 = 1 # 1A, normalized!
obs = [1,1,1]
#obs_points = create_uniform_sphere(obs_distance=wavelength*10, num_points=3000, plot=False)
points = generate_meander_antenna(lambda_g, d, s, L, w, N=4, num_points=num_points, plot=False) # generate the segmetns of the meander line

points_3d = generate_3d_structure(points_2D=points, wire_height=wire_height, plot=False)  # plot=True must use plt.show(block=False)

s_array, s_hat_array, r_array, h_list, h_hat_array, center_points = observer_relation_locally(obs, points_3d, plot=False)





boundaries = np.cumsum(h_list)
positions = np.insert(boundaries, 0, 0)
center_points_array = []

# compute segment centers
for i in range(len(positions)):
    center_point = (positions[i-1] + positions[i]) / 2
    center_points_array.append(center_point)

antenna_start = 0
antenna_end = np.max(boundaries)

sinus_wave_length = antenna_end - antenna_start
sinus_freq = (3.0 * 10**8) / sinus_wave_length
amplitude = 1  # normalized

x_vals = np.linspace(antenna_start, antenna_end, 1000)
y_vals = amplitude * np.sin(2 * np.pi * x_vals / sinus_wave_length)

# plot total line
plt.hlines(0, 0, positions[-1], colors="black", linewidth=2)

# plot vertical ticks at boundaries
for x in positions:
    plt.vlines(x, -0.2, 0.2, colors="red")

# plot scatter points at centers
plt.scatter(center_points_array, [0]*len(center_points_array), color="blue")

# plot sinusoidal wave
plt.plot(x_vals, y_vals, color="green")

# formatting
plt.title("Segments along total length with sine wave")
plt.xlabel("Length")
plt.yticks([])
plt.show()




''''
def create_current_distrubtion(h_array, center_points):
    #start by reducing  

    return None




# Compute fields
E_total_array, H_total_array = compute_fields_for_points(obs_points, points_3d, eta, k, I0)

import numpy as np
import matplotlib.pyplot as plt

# Compute magnitude
E_magnitude = np.linalg.norm(E_total_array, axis=1)

# Normalize observation points (radial directions only)
obs_norm = obs_points / np.linalg.norm(obs_points, axis=1)[:, None]

# Scale the directions by magnitude
scaled_points = obs_norm * E_magnitude[:, None]

import numpy as np
import matplotlib.pyplot as plt

# Compute magnitude
E_magnitude = np.linalg.norm(E_total_array, axis=1)

# Normalize observation points (radial directions only)
obs_norm = obs_points / np.linalg.norm(obs_points, axis=1)[:, None]

# Scale the directions by magnitude
scaled_points = obs_norm * E_magnitude[:, None]


def plot_magnitude_radiation(observation_points, field_array, dB=True, label=''):
    # Compute field magnitude
    if dB:
        E_magnitude = 20 * np.log10(np.linalg.norm(field_array, axis=1))
    else:
        E_magnitude = np.linalg.norm(field_array, axis=1)

    # Normalize observation vectors
    obs_norm = observation_points / np.linalg.norm(observation_points, axis=1)[:, None]
    scaled_points = obs_norm * E_magnitude[:, None]

    # Setup plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    p = ax.scatter(
        scaled_points[:, 0], scaled_points[:, 1], scaled_points[:, 2],
        c=E_magnitude, cmap='turbo'
    )

    # Add colorbar + labels
    if dB:
        fig.colorbar(p, label=f'Radiated {label}-Field in dB')
        ax.set_title(f'Radiated {label}-Field in dB')
    else:
        fig.colorbar(p, label=f'Radiated {label}-Field')
        ax.set_title(f'Radiated {label}-Field')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()



plot_magnitude_radiation(obs_points, E_total_array, dB=True, label='E')
'''
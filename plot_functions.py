import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from segmentation import generate_meander_antenna, generate_3d_structure
from local_evaluation import plot_vectors, observer_relation_globally, observer_relation_locally, current_distribution
from evaluate_fields import compute_fields_for_points, create_uniform_sphere, calculate_directivity
import numpy as np
from scipy.interpolate import griddata
from helper_functions import convert_cartesian_to_spherical, calculate_directivity


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


def plot_magnitude_radiation(observation_points, total_magnitude, dB=True, label=''):

    obs_norm = observation_points / np.linalg.norm(observation_points, axis=1)[:, None]
    scaled_points = obs_norm * total_magnitude[:, None]

    if dB:
        E_color = 20 * np.log10(total_magnitude + 1e-20)  # avoid log(0)
    else:
        E_color = total_magnitude

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    p = ax.scatter(
        scaled_points[:, 0], scaled_points[:, 1], scaled_points[:, 2],
        c=E_color, cmap='turbo'
    )

    # Colorbar and titles
    if dB:
        fig.colorbar(p, label=f'Radiated {label}-Field [dB]')
        ax.set_title(f'Radiated {label}-Field in dB')
    else:
        fig.colorbar(p, label=f'Radiated {label}-Field [Mag]')
        ax.set_title(f'Radiated {label}-Field')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

    # Return the linear magnitude array
    return None

def half_power_beam_width(E_magnitude, observation_points):
    E_magnitude_db = 20 * np.log10(E_magnitude)
    max_val = np.max(E_magnitude_db)

    for i in range(len(E_magnitude_db)):
        if E_magnitude_db[i] <= max_val - 3:
            x, y, z = observation_points[i]
            r = np.sqrt(x**2 + y**2 + z**2)
            theta = np.degrees(np.arccos(z / r))  
            phi = np.degrees(np.arctan2(y, x))    
            return theta, phi  

    return None, None
    


lambda_g = 179e-3
d = 0.16 * lambda_g
s = 0.42 * lambda_g
L = 0.7 * lambda_g
w = 0.05 * lambda_g
wire_height = 3.2e-3
num_points = 200
Freq_of_interest = 1060e6 #1060MHz
wavelength = 3.0 * 10**8 / Freq_of_interest
eta = 377 #Ohms.. intrinsic impedance, not sure of this
k = 2 * np.pi / lambda_g

obs_point = np.array([[3, 3, 3]])
obs_points = create_uniform_sphere(obs_distance=wavelength*10, num_points=1000, plot=False)
points = generate_meander_antenna(lambda_g, d, s, L, w, N=4, num_points=num_points, plot=False) 
points_3d = generate_3d_structure(points_2D=points, wire_height=wire_height, plot=False) 
s_array, s_hat_array, r_array, h_list, h_hat_array, center_points = observer_relation_locally(obs_point, points_3d, plot=False)
_, I0 = current_distribution(h_list, wavelength, amplitude=1.0, plot=False)



E_total_array, H_total_array, E_total_mag, H_total_mag = compute_fields_for_points(obs_points, points_3d, eta, k, I0)

#plot_magnitude_radiation(obs_points, E_total_mag, dB=False, label='E')
E_total_spherical, theta, phi = convert_cartesian_to_spherical(E_total_array, obs_points)
H_total_spherical, dummy1, dummy2 = convert_cartesian_to_spherical(H_total_array, obs_points)
print(theta)

D0 = calculate_directivity(E_total_spherical, H_total_spherical, theta, phi)


#theta_E, phi_E = half_power_beam_width(E_total_mag, obs_points)

'''
theta_H, phi_H = half_power_beam_width(H_total_mag, obs_points)


G = 32400 / (theta_E * theta_H)
D = 41253 / (theta_E * theta_H)
print(G)
print(D)

'''
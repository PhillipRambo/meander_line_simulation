import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from segmentation import generate_meander_antenna, generate_3d_structure
from local_evaluation import plot_vectors, observer_relation_globally, observer_relation_locally, current_distribution, segment_integrals
from evaluate_fields import compute_fields_for_points, create_uniform_sphere, calculate_directivity
import numpy as np
from scipy.interpolate import griddata
from helper_functions import convert_cartesian_to_spherical, calculate_directivity
from plot_2d import create_azimuthal_cutout, plot_spherical_field_polar


def create_uniform_sphere(obs_distance, num_points, plot=True):
    indices = np.arange(0, num_points, dtype=float) + 0.5
    phi = 2 * np.pi * indices / ((1 + np.sqrt(5)) / 2)
    theta = np.arccos(1 - 2*indices/num_points)

    X = obs_distance * np.sin(theta) * np.cos(phi)
    Y = obs_distance * np.sin(theta) * np.sin(phi)
    Z = obs_distance * np.cos(theta)
    X = np.concatenate([X, [0, 0]])
    Y = np.concatenate([Y, [0, 0]])
    Z = np.concatenate([Z, [obs_distance, -obs_distance]])
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
    # Normalize direction vectors
    obs_norm = observation_points / np.linalg.norm(observation_points, axis=1)[:, np.newaxis]
    
    # Scale by magnitude
    scaled_points = obs_norm * total_magnitude[:, np.newaxis]

    # Color values
    if dB:
        E_color = 20 * np.log10(total_magnitude)
    else:
        E_color = total_magnitude

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    p = ax.scatter(
        scaled_points[:, 0], scaled_points[:, 1], scaled_points[:, 2],
        c=E_color, cmap='turbo'
    )

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
    
def plot_e_field_heatmap_dB(E_spherical, theta, phi, theta_points=180, phi_points=360):
    """
    Plot a 2D heatmap of the E-field magnitude in dB on a theta-phi grid.

    Parameters:
    -----------
    E_spherical : np.ndarray
        Array of E-field vectors in spherical coordinates (N x 3).
    theta : np.ndarray
        Array of theta angles for each observation point (radians).
    phi : np.ndarray
        Array of phi angles for each observation point (radians).
    theta_points : int
        Number of points in the theta grid (default 180).
    phi_points : int
        Number of points in the phi grid (default 360).
    """
    # Compute magnitude
    E_magnitude = np.linalg.norm(E_spherical, axis=1)

    # Map phi to [0, 2*pi]
    phi_mod = np.mod(phi, 2*np.pi)

    # Create uniform grid
    theta_grid = np.linspace(theta.min(), theta.max(), theta_points)
    phi_grid = np.linspace(0, 2*np.pi, phi_points)
    Theta, Phi = np.meshgrid(theta_grid, phi_grid)

    # Interpolate scattered data onto the grid
    E_grid = griddata(
        points=(theta, phi_mod),
        values=E_magnitude,
        xi=(Theta, Phi),
        method='linear'
    )

    # Convert to dB (avoid log(0))
    E_dB = 20 * np.log10(E_grid + 1e-12)

    # Plot heatmap
    plt.figure(figsize=(8,6))
    plt.pcolormesh(Phi*180/np.pi, Theta*180/np.pi, E_dB, shading='auto', cmap='viridis')
    plt.colorbar(label='|E| [dB]')
    plt.xlabel('Phi [deg]')
    plt.ylabel('Theta [deg]')
    plt.title('E-field magnitude heatmap [dB]')
    plt.show()




lambda_g = 15.13e-2
d = 0.16 * lambda_g
s = 0.33 * lambda_g
L = 0.7 * lambda_g
w = 0.06 * lambda_g
wire_height = 3.2e-3
num_points = 100
Freq_of_interest = 1060e6 #1060MHz
wavelength = 3.0 * 10**8 / Freq_of_interest
eta = 377 #Ohms.. intrinsic impedance, not sure of this
k = 2 * np.pi / lambda_g

obs_point = np.array([[0.1, 0.1, 0.1]])
#obs_points = create_uniform_sphere(obs_distance=wavelength*10, num_points=500, plot=False)
phi_true = 0
obs_points, theta_true = create_azimuthal_cutout(r=wavelength*10, phi=phi_true, points=100, plot=True)
points = generate_meander_antenna(lambda_g, d, s, L, w, N=4, num_points=num_points, plot=False) 
points_3d = generate_3d_structure(points_2D=points, wire_height=wire_height, plot=False) 
s_array, s_hat_array, r_array, h_list, h_hat_array, center_points = observer_relation_locally(obs_point, points_3d, plot=False)
_, I0 = current_distribution(h_list, wavelength, amplitude=1.0, plot=False)
I_seg = segment_integrals(h_list, lambda_g, amplitude=1.0, L=None, plot=False)
I = 1

E_total_array, H_total_array = compute_fields_for_points(obs_points, points_3d, eta, k, I_seg)
plot_magnitude_radiation(obs_points, np.linalg.norm(E_total_array, axis=1), dB=True, label='E')

E_total_spherical, theta, phi = convert_cartesian_to_spherical(E_total_array, obs_points)
plot_spherical_field_polar(E_total_spherical, theta_true, phi_true, dB=True, label='E-field')

'''
H_total_spherical, dummy1, dummy2 = convert_cartesian_to_spherical(H_total_array, obs_points)
plot_e_field_heatmap_dB(E_total_spherical, theta, phi)

theta_E, phi_E = half_power_beam_width(np.linalg.norm(E_total_array, axis=1), obs_points)
theta_H, phi_H = half_power_beam_width(np.linalg.norm(H_total_array, axis=1), obs_points)

half_power_beamwidth_product = (180-theta_E*2) * (180-theta_H*2) 


G = 32400 / half_power_beamwidth_product
D = 41253 / half_power_beamwidth_product
e_r = G/D #radiation effeciency
print(f'G = {G}')
print(f'D = {D}')



D0 = calculate_directivity(E_total_spherical, H_total_spherical, theta, phi)

print(f'D0:{D0}')

'''

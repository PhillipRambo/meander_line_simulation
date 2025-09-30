from local_evaluation import plot_vectors, observer_relation_globally, observer_relation_locally
import matplotlib.pyplot as plt
import numpy as np

def compute_electrical_and_magnetic_field(
    n, k, I0, h_array, r_array, phi_hat, r_hat, theta_hat, sin_theta_array, cos_theta_array, s_hat, magnitude=True
):
    N = len(r_array)
    E_spherical_array = []
    H_spherical_array = []
    E_magnitude_array = []
    H_magnitude_array = []

    for i in range(N):
        # --- Spherical field components ---
        E_r = n * (I0[i] * h_array[i] * cos_theta_array[i]) / (2 * np.pi * r_array[i]**2) * (1 + 1/(1j * k * r_array[i])) * np.exp(-1j * k * r_array[i])
        E_theta = 1j * n * (k * I0[i] * h_array[i] * sin_theta_array[i]) / (4 * np.pi * r_array[i]) * (1 + 1/(1j * k * r_array[i]) - 1/(k * r_array[i])**2) * np.exp(-1j * k * r_array[i])
        E_phi = 0
        H_r = 0
        H_theta = 0
        H_phi = 1j * (k * I0[i] * h_array[i]  * sin_theta_array[i]) / (4 * np.pi * r_array[i]) * (1 + 1/(1j * k * r_array[i])) * np.exp(-1j * k * r_array[i])

        # --- making a 3D vector for the E and H field
        E_spherical_local = np.array([E_r, E_theta, E_phi])
        H_spherical_local = np.array([H_r, H_theta, H_phi])

        # Converting into a single 3D vector for a single segment to observation point
        E_spherical_global = E_spherical_local[0]*r_hat[i] + E_spherical_local[1]*theta_hat[i] + E_spherical_local[2]*phi_hat[i]
        H_spherical_global = H_spherical_local[0]*r_hat[i] + H_spherical_local[1]*theta_hat[i] + H_spherical_local[2]*phi_hat[i]

        E_magnitude_try = np.linalg.norm(E_spherical_global) 
        H_magnitude_try = np.linalg.norm(H_spherical_global)  

        E_spherical_array.append(E_spherical_global)
        H_spherical_array.append(H_spherical_global)
        
        E_magnitude_array.append(E_magnitude_try)
        H_magnitude_array.append(H_magnitude_try)


    E_spherical_array = np.array(E_spherical_array)
    H_spherical_array = np.array(H_spherical_array)
    E_magnitude_array = np.array(E_magnitude_array)
    H_magnitude_array = np.array(H_magnitude_array)

    E_total_mag = np.sum(E_magnitude_array)
    H_total_mag = np.sum(E_magnitude_array)

    E_total_spherical = np.sum(E_spherical_array, axis=0)
    H_total_spherical = np.sum(H_spherical_array, axis=0)

    return {
        "E_total_spherical": E_total_spherical,
        "H_total_spherical": H_total_spherical,
        "E_total_mag": E_total_mag,
        "H_total_mag": H_total_mag,
    }




def compute_fields_for_points(observer_points, points_3d, eta, k, I0):
    E_total = []
    H_total = []
    E_total_mag = []
    H_total_mag = []

    for obs in observer_points:
        s_array, s_hat_array, r_array, h_list, h_hat_array, center_points = observer_relation_locally(obs, points_3d, plot=False)
        phi_hat, r_hat, theta_hat, sin_theta, cos_theta = observer_relation_globally(s_hat_array, h_hat_array)
        
        fields = compute_electrical_and_magnetic_field(
            eta, k, I0, h_list, r_array, phi_hat, r_hat, theta_hat, sin_theta, cos_theta, s_hat_array, magnitude=True
        )
    

        E_total.append(fields["E_total_spherical"])
        H_total.append(fields["H_total_spherical"])
        E_total_mag.append(fields["E_total_mag"])
        H_total_mag.append(fields["H_total_mag"])

                              
    E_total_array = np.array(E_total)
    H_total_array = np.array(H_total)
    E_total_mag_array = np.array(E_total_mag)
    H_total_mag_array = np.array(H_total_mag)

    return E_total_array, H_total_array, E_total_mag_array, H_total_mag_array


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


def calculate_directivity(E_field, H_field, observationpoints):
    x, y, z = observationpoints[:, 0], observationpoints[:, 1], observationpoints[:, 2]
    r = np.linalg.norm(observationpoints, axis=1)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)

    dtheta = np.abs(theta[1] - theta[0])
    dphi = np.abs(phi[1] - phi[0])

    S_denom = np.real(np.linalg.norm(np.cross(E_field, np.conj(H_field)), axis=1))

    S = 0.5 * np.real(np.linalg.norm(np.cross(E_field, np.conj(H_field)), axis=1))

    # Sum over surface with sin(theta)
    S_sum = np.sum(S_denom * np.sin(theta))

    # Peak Poynting magnitude
    S_max = np.max(S)

    # Directivity according to equation 1.6
    D0 = (8 * np.pi / (dphi * dtheta)) * (S_max / S_sum)
    return D0

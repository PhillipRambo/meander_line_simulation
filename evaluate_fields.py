from local_evaluation import plot_vectors, observer_relation_globally, observer_relation_locally
import matplotlib.pyplot as plt
import numpy as np

def compute_electrical_and_magnetic_field(
    n, k, I0, h_array, r_array, phi_hat, r_hat, theta_hat, sin_theta_array, cos_theta_array, s_hat, magnitude=True
):
    N = len(r_array)

    E_r_array = []
    E_theta_array = []
    H_phi_array = []
    E_cartesian_array = []
    H_cartesian_array = []
    E_spherical_array = []
    H_spherical_array = []

    for i in range(N):
        s_x, s_y, s_z = s_hat[i]
        theta = np.arccos(cos_theta_array[i])
        phi = np.arctan2(s_y, s_x)
        # --- Spherical field components ---
        E_r = n * (I0[i] * h_array[i] * cos_theta_array[i]) / (2 * np.pi * r_array[i]**2) * (1 + 1/(1j * k * r_array[i])) * np.exp(-1j * k * r_array[i])
        E_theta = 1j * n * (k * I0[i] * h_array[i] * sin_theta_array[i]) / (4 * np.pi * r_array[i]) * (1 + 1/(1j * k * r_array[i]) - 1/(k * r_array[i])**2) * np.exp(-1j * k * r_array[i])
        E_phi = 0
        H_r = 0
        H_theta = 0
        H_phi = 1j * (k * I0[i] * h_array[i] * sin_theta_array[i]) / (4 * np.pi * r_array[i]) * (1 + 1/(1j * k * r_array[i])) * np.exp(-1j * k * r_array[i])

        # --- Conversion to Cartesian ---
        conv_matrix = np.array([
            [np.sin(theta)*np.cos(phi), np.cos(theta)*np.cos(phi), -np.sin(phi)],
            [np.sin(theta)*np.sin(phi), np.cos(theta)*np.sin(phi),  np.cos(phi)],
            [np.cos(theta),             -np.sin(theta),             0]
        ], dtype=complex)

        E_cart_local = conv_matrix @ np.array([E_r, E_theta, E_phi])
        H_cart_local = conv_matrix @ np.array([H_r, H_theta, H_phi])

        E_spherical_local = np.array([E_r, E_theta, E_phi])
        H_spherical_local = np.array([H_r, H_theta, H_phi])

        E_cart_global = E_cart_local[0]*r_hat[i] + E_cart_local[1]*theta_hat[i] + E_cart_local[2]*phi_hat[i]
        H_cart_global = H_cart_local[0]*r_hat[i] + H_cart_local[1]*theta_hat[i] + H_cart_local[2]*phi_hat[i]

        E_spherical_global = E_spherical_local[0]*r_hat[i] + E_spherical_local[1]*theta_hat[i] + E_spherical_local[2]*phi_hat[i]
        H_spherical_global = H_spherical_local[0]*r_hat[i] + H_spherical_local[1]*theta_hat[i] + H_spherical_local[2]*phi_hat[i]

    
        # Store per-segment values
        E_r_array.append(abs(E_r) if magnitude else E_r)
        E_theta_array.append(abs(E_theta) if magnitude else E_theta)
        H_phi_array.append(abs(H_phi) if magnitude else H_phi)
        E_cartesian_array.append(abs(E_cart_global) if magnitude else E_cart_global)
        H_cartesian_array.append(abs(H_cart_global) if magnitude else H_cart_global)
        E_spherical_array.append(abs(E_spherical_global) if magnitude else E_spherical_global)
        H_spherical_array.append(abs(H_spherical_global) if magnitude else H_spherical_global)

    E_r_array = np.array(E_r_array)
    E_theta_array = np.array(E_theta_array)
    H_phi_array = np.array(H_phi_array)
    E_cartesian_array = np.array(E_cartesian_array)
    H_cartesian_array = np.array(H_cartesian_array)
    E_spherical_array = np.array(E_spherical_array)
    H_spherical_array = np.array(H_spherical_array)


    E_total_cart = np.sum(E_cartesian_array, axis=0)
    H_total_cart = np.sum(H_cartesian_array, axis=0)
    E_total_spherical = np.sum(E_spherical_array, axis=0)
    H_total_spherical = np.sum(H_spherical_array, axis=0)

    return {
        "E_spherical_segments": np.column_stack((E_r_array, E_theta_array, np.zeros(N))),
        "H_spherical_segments": np.column_stack((np.zeros(N, dtype=complex), np.zeros(N, dtype=complex), H_phi_array)),
        "E_cartesian_segments": E_cartesian_array,
        "H_cartesian_segments": H_cartesian_array,
        "E_total_cartesian": E_total_cart,
        "H_total_cartesian": H_total_cart,
        "E_total_spherical": E_total_spherical,
        "H_total_spherical": H_total_spherical
    }




def compute_fields_for_points(observer_points, points_3d, eta, k, I0):
    E_total = []
    H_total = []

    for obs in observer_points:
        s_array, s_hat_array, r_array, h_list, h_hat_array, center_points = observer_relation_locally(obs, points_3d, plot=False)
        phi_hat, r_hat, theta_hat, sin_theta, cos_theta = observer_relation_globally(s_hat_array, h_hat_array)
        

        fields = compute_electrical_and_magnetic_field(
            eta, k, I0, h_list, r_array, phi_hat, r_hat, theta_hat, sin_theta, cos_theta, s_hat_array, magnitude=True
        )
    

        #E_total.append(fields["E_total_cartesian"])
        #H_total.append(fields["H_total_cartesian"])
        E_total.append(fields["E_total_spherical"])
        H_total.append(fields["H_total_spherical"])
                              
    E_total_array = np.array(E_total)
    H_total_array = np.array(H_total)


    return E_total_array, H_total_array



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

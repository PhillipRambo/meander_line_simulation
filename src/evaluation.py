from .local_eval import observer_relation_globally, observer_relation_locally
import matplotlib.pyplot as plt
import numpy as np

def compute_electrical_and_magnetic_field(
    n, k, I0, h_array, r_array, phi_hat, r_hat, theta_hat, sin_theta_array, cos_theta_array, s_hat, magnitude=True
):
    N = len(r_array)
    E_cartesian_array = []
    H_cartesian_array = []

    for i in range(N):
        # --- Spherical field components ---
        E_r = n * (I0[i] * cos_theta_array[i]) / (2 * np.pi * r_array[i]**2) * (1 + 1/(1j * k * r_array[i])) * np.exp(-1j * k * r_array[i])
        E_theta = 1j * n * (k * I0[i] * sin_theta_array[i]) / (4 * np.pi * r_array[i]) * (1 + 1/(1j * k * r_array[i]) - 1/(k * r_array[i])**2) * np.exp(-1j * k * r_array[i])
        E_phi = 0
        H_r = 0
        H_theta = 0
        H_phi = 1j * (k * I0[i] * sin_theta_array[i]) / (4 * np.pi * r_array[i]) * (1 + 1/(1j * k * r_array[i])) * np.exp(-1j * k * r_array[i])

        # --- making a 3D vector for the E and H field
        E_spherical_local = np.array([E_r, E_theta, E_phi])
        H_spherical_local = np.array([H_r, H_theta, H_phi])

        # Converting into a single 3D vector for a single segment to observation point
        E_cartesian_global = E_spherical_local[0]*r_hat[i] + E_spherical_local[1]*theta_hat[i] + E_spherical_local[2]*phi_hat[i]
        H_cartesian_global = H_spherical_local[0]*r_hat[i] + H_spherical_local[1]*theta_hat[i] + H_spherical_local[2]*phi_hat[i]

        E_cartesian_array.append(E_cartesian_global)
        H_cartesian_array.append(H_cartesian_global)
        
    E_cartesian_array = np.array(E_cartesian_array)
    H_cartesian_array = np.array(H_cartesian_array)


    E_total_cartesian = np.sum(E_cartesian_array, axis=0)
    H_total_cartesian = np.sum(H_cartesian_array, axis=0)

    return {
        "E_total_cartesian": E_total_cartesian,
        "H_total_cartesian": H_total_cartesian,
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

        E_total.append(fields["E_total_cartesian"])
        H_total.append(fields["H_total_cartesian"])
                              
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
        ax.set_title('Observation Points on Uniform Sphere')  # <-- add this line

        plt.show()
        return np.stack([X, Y, Z], axis=1)
    else:
        return np.stack([X, Y, Z], axis=1)


def create_azimuthal_cutout(r, phi, points=180,plot=False):

    theta = np.linspace(0, 2*np.pi, points)  
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    eps = 1e-12
    if np.isclose(np.cos(phi), 0, atol=eps):  
        x = np.zeros_like(theta)
    elif np.isclose(np.sin(phi), 0, atol=eps):  
        y = np.zeros_like(theta)
    
    obs_points = np.stack((x, y, z), axis=-1)
    
    if plot:
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(obs_points[:,0], obs_points[:,1], obs_points[:,2], c='r', s=20)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Observation Points on plane')  # <-- add this line
        ax.set_box_aspect([1,1,1])
        plt.show()
    
    return obs_points, theta


'''

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

'''
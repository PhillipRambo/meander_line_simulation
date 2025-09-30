import numpy as np

def convert_cartesian_to_spherical(field_vector, observation_points):
    E_array = []
    theta_list = []
    phi_list = []
    
    for i in range(len(observation_points)):
        s_x, s_y, s_z = observation_points[i] 
        r = np.sqrt(s_x**2 + s_y**2 + s_z**2)
        theta = np.arccos(s_z/r)
        phi = np.arctan2(s_y, s_x)

        conv_matrix = np.array([
            [np.sin(theta)*np.cos(phi), np.cos(theta)*np.cos(phi), -np.sin(phi)],
            [np.sin(theta)*np.sin(phi), np.cos(theta)*np.sin(phi),  np.cos(phi)],
            [np.cos(theta), -np.sin(theta), 0]
        ], dtype=complex)
        
        E_cart = conv_matrix @ field_vector[i]
        E_array.append(E_cart)
        theta_list.append(theta)
        phi_list.append(phi)

    E_spherical = np.array(E_array)
    theta_array = np.array(theta_list)
    phi_array = np.array(phi_list)

    
    return E_spherical, theta_array, phi_array


def calculate_directivity(E_field, H_field, theta, phi):
    E_theta = E_field[:, 1]
    E_phi   = E_field[:, 2]
    H_theta = H_field[:, 1]
    H_phi   = H_field[:, 2]
    for i in range(len(E_theta)):
        S_r = 0.5 * np.real(E_theta[i] * np.conj(H_phi[i]) - E_phi[i] * np.conj(H_theta[i]))
        print((E_theta[i] * np.conj(H_phi[i]) - E_phi[i] * np.conj(H_theta[i])))
    return None
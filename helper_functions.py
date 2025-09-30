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
    S = 0.5 * np.real(np.cross(E_field, np.conj(H_field)))
    S_max = np.linalg.norm(np.max(S))
    denom = np.sum(S * np.sin(theta)[:, np.newaxis])
    dtheta = theta[1] - theta[0]
    dphi   = phi[1] - phi[0]
    D0 = (8 * np.pi / (np.abs(dphi) * np.abs(dtheta))) * (S_max / denom)

    return D0
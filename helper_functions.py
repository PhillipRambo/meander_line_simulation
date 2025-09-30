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
    S_r = []
    S_denom = []

    for i in range(len(E_field)):

        E_theta = E_field[:,1][i]
        H_theta = H_field[:,1][i]

        E_phi = E_field[:,2][i]
        H_phi = H_field[:,2][i]

        s_r = 0.5 * np.real(E_theta * np.conj(H_phi) - E_phi * np.conj(H_theta))
        s_denom = np.real(np.cross(E_field[i],np.conj(H_field[i]))) * np.sin(theta[i])
        
        S_r.append(s_r)
        S_denom.append(s_denom)
    
    S_r_array = np.array(S_r)
    S_denom_array = np.array(S_denom)


    s_r_max = np.max(S_r_array)
    s_r_max_mag = np.linalg.norm(s_r_max)
    S_sum = np.sum(S_denom_array, axis = 0)

    theta_delta = theta[1] - theta[0]
    phi_delta = phi[1] - phi[0]

    D0 =( 8 * np.pi / (phi_delta * theta_delta) ) * ( s_r_max_mag / S_sum )

    return D0
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata




def plot_magnitude_radiation(observation_points, total_magnitude, dB=True, label=''):
    obs_norm = observation_points / np.linalg.norm(observation_points, axis=1)[:, np.newaxis]
    
    scaled_points = obs_norm * total_magnitude[:, np.newaxis]

    if dB:
        E_color = 20 * np.log10(total_magnitude)
    else:
        E_color = total_magnitude

    
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

    E_magnitude = np.linalg.norm(E_spherical, axis=1)

    phi_mod = np.mod(phi, 2*np.pi)

    theta_grid = np.linspace(theta.min(), theta.max(), theta_points)
    phi_grid = np.linspace(0, 2*np.pi, phi_points)
    Theta, Phi = np.meshgrid(theta_grid, phi_grid)

    E_grid = griddata(
        points=(theta, phi_mod),
        values=E_magnitude,
        xi=(Theta, Phi),
        method='linear'
    )

    E_dB = 20 * np.log10(E_grid + 1e-12)

    plt.figure(figsize=(8,6))
    plt.pcolormesh(Phi*180/np.pi, Theta*180/np.pi, E_dB, shading='auto', cmap='viridis')
    plt.colorbar(label='|E| [dB]')
    plt.xlabel('Phi [deg]')
    plt.ylabel('Theta [deg]')
    plt.title('E-field magnitude heatmap [dB]')
    plt.show()



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
        ax.set_box_aspect([1,1,1])
        plt.show()
    
    return obs_points, theta




def plot_spherical_field_polar(E_spherical, theta, phi, dB=True, label='E-field'):

    E_mag = np.linalg.norm(E_spherical, axis=1)

    if dB:
        E_mag = 20 * np.log10(E_mag / np.max(E_mag))

    phi_deg = np.degrees(phi)

 
    plt.figure(figsize=(6,6))
    ax = plt.subplot(111, polar=True)
    ax.plot(theta, E_mag, label=label)

    ax.set_theta_zero_location("N")  # 0 deg at top
    ax.set_theta_direction(-1)       # clockwise

    ax.set_title(f"{label} magnitude normalized @ (phi = {phi_deg:.1f}°)")
    ax.set_xlabel("Theta (°)", labelpad=20)  
    ax.grid(True)
    plt.legend()
    plt.show()





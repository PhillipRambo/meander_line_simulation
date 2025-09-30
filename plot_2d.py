import numpy as np
import matplotlib.pyplot as plt

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
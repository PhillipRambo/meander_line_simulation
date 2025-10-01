from src import *
import config as cfg
import numpy as np

# -------------------- PLOTTING FLAGS --------------------
PLOT_STRUCTURE = True
PLOT_CURRENT = True
PLOT_FIELDS = True
PLOT_HEATMAP = True
PLOT_POLAR = True

def main():
    ################# ------------------- Create the Structure ------------------- #################

    obs_points_sphere = create_uniform_sphere(
        obs_distance=cfg.wavelength*10,
        num_points=cfg.num_obs_points,
        plot=PLOT_STRUCTURE
    )

    obs_points_azimuthal, theta_true = create_azimuthal_cutout(
        r=cfg.wavelength*10,
        phi=np.pi/2,
        points=100,
        plot=PLOT_STRUCTURE
    )

    points_2d = generate_meander_antenna(
        cfg.lambda_g, cfg.d, cfg.s, cfg.L, cfg.w,
        N=4, num_points=cfg.num_points,
        plot=PLOT_STRUCTURE
    )

    points_3d = generate_3d_structure(
        points_2d,
        wire_height=cfg.wire_height,
        plot=PLOT_STRUCTURE
    )

    ################# ------------------- Make a Current Distribution over the Antenna ------------------- #################

    _, _, _, h_list, _, _ = observer_relation_locally(cfg.obs_point, points_3d, plot=PLOT_CURRENT)

    I_seg = segment_integrals(h_list, cfg.lambda_g, amplitude=1.0, L=None, plot=PLOT_CURRENT)

    ################# ------------------- Compute the Electrical Field over Sphere and In plane ------------------- #################

    E_total_array_sphere, H_total_array_sphere = compute_fields_for_points(
        obs_points_sphere, points_3d, cfg.eta, cfg.k, I_seg
    )

    E_total_array_azimuthal, H_total_array_azimuthal = compute_fields_for_points(
        obs_points_azimuthal, points_3d, cfg.eta, cfg.k, I_seg
    )

    ################# ------------------- Conversion functions ------------------- #################

    E_total_spherical, theta, phi = convert_cartesian_to_spherical(E_total_array_sphere, obs_points_sphere)
    H_total_spherical, _, _ = convert_cartesian_to_spherical(H_total_array_sphere, obs_points_sphere)
    E_total_spherica_azimuthal, theta_azimuthal, phi_azimuthal = convert_cartesian_to_spherical(
        E_total_array_azimuthal, obs_points_azimuthal
    )

    ################# ------------------- Plotting ------------------- #################

    if PLOT_FIELDS:
        plot_magnitude_radiation(obs_points_sphere, np.linalg.norm(E_total_array_sphere, axis=1), dB=True, label='E')
        plot_magnitude_radiation(obs_points_azimuthal, np.linalg.norm(E_total_array_azimuthal, axis=1), dB=True, label='E')

    if PLOT_HEATMAP:
        plot_e_field_heatmap_dB(E_total_spherical, theta, phi)

    if PLOT_POLAR:
        plot_spherical_field_polar(E_total_spherica_azimuthal, theta_true, np.pi/2, dB=True, label='E-field')

    ################# ------------------- Calculating Gain and Directivity ------------------- #################

    theta_E, phi_E = half_power_beam_width(np.linalg.norm(E_total_array_sphere, axis=1), obs_points_sphere)
    theta_H, phi_H = half_power_beam_width(np.linalg.norm(H_total_array_sphere, axis=1), obs_points_sphere)

    half_power_beamwidth_product = (180-theta_E*2) * (180-theta_H*2)
    G = 32400 / half_power_beamwidth_product
    D = 41253 / half_power_beamwidth_product
    e_r = G / D  # Radiation efficiency

    print(f'G = {G}')
    print(f'D = {D}')
    print(f'radiation efficiency: {e_r * 100} %')
    D0 = calculate_directivity(E_total_spherical, H_total_spherical, theta, phi)
    print(f'D0: {D0}')

if __name__ == "__main__":
    main()

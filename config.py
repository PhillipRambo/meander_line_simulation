import numpy as np

eta = 377                
c = 3e8                   

# Frequency
Freq_of_interest = 1060e6
wavelength = c / Freq_of_interest
k = 2 * np.pi / (0.1513)  

# Antenna geometry
lambda_g = 0.1513
d = 0.16 * lambda_g
s = 0.33 * lambda_g
L = 0.7 * lambda_g
w = 0.06 * lambda_g
wire_height = 3.2e-3
num_points = 100

# Observation points
obs_point = np.array([[0.1, 0.1, 0.1]])
num_obs_points = 1000
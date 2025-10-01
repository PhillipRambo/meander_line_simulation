# Meander Line Simulation

This repository contains code for simulating, analyzing, and visualizing the radiation patterns of antennas for an RF course assignment.

## Repository Structure

```
meander_line_simulation/
├── LICENSE
├── README.md
├── config.py               # Constants and parameters
├── main.py                 # Main script to run the assignment
├── src/                    # Source code package
│   ├── __init__.py
│   ├── helpers.py          # Helper functions
│   ├── evaluation.py       # Field calculations
│   ├── plotting.py         # Plotting functions
│   ├── segmentation.py     # Antenna geometry generation
│   └── local_eval.py       # Local evaluation functions
└── requirements.txt        # Required Python packages
```

## Installation

1. Clone the repository:
```
git clone https://github.com/PhillipRambo/meander_line_simulation.git
```
```
cd meander_line_simulation
```
2. Install the required packages:

```
pip install -r requirements.txt
```
## Usage

Run the main script:
```
python main.py
```

⚠️ **NOTE:** In the `main.py` file, you have flags to select what you want to plot.

## Configuration

All constants and parameters are stored in config.py. You can adjust:

- Antenna dimensions (lambda_g, d, s, L, w, wire_height)
- Observation points
- Frequency of interest (Freq_of_interest)
- Other simulation parameters

## Modules

- src/helpers.py: Utility functions such as coordinate conversions and directivity calculation
- src/evaluation.py: Functions to compute E and H fields
- src/plotting.py: Functions for plotting radiation patterns, heatmaps, and 3D visualizations
- src/segmentation.py: Functions to generate antenna geometries
- src/local_eval.py: Local evaluation functions for currents, segments, and observer relations


# src/__init__.py
from .helpers import convert_cartesian_to_spherical, calculate_directivity
from .evaluation import compute_electrical_and_magnetic_field, compute_fields_for_points, create_uniform_sphere, create_azimuthal_cutout
from .plotting import plot_magnitude_radiation, plot_e_field_heatmap_dB, plot_spherical_field_polar, half_power_beam_width
from .segmentation import generate_meander_antenna, generate_3d_structure
from .local_eval import observer_relation_locally , segment_integrals, observer_relation_globally

__all__ = [
    "convert_cartesian_to_spherical",
    "calculate_directivity",
    "compute_electrical_and_magnetic_field",
    "compute_fields_for_points",
    "create_uniform_sphere",
    "plot_magnitude_radiation",
    "plot_e_field_heatmap_dB",
    "create_azimuthal_cutout",
    "plot_spherical_field_polar",
    "half_power_beam_width",
    "generate_meander_antenna",
    "generate_3d_structure",
    "observer_relation_locally",
    "observer_relation_globally",
    "segment_integrals",
]


"""RADON: Optimal Tomographic Probing for Neural Curvature.

A standalone second-order optimizer and curvature estimation framework
built on split-exact curvature theory and coded directional projections.
"""

from .optimizer import Radon
from .probes import ProbeGenerator, code_tensor, hadamard
from .split import fisher_diag_sample, residual_probe

__version__ = "1.0.0"
__all__ = [
    "ProbeGenerator",
    "Radon",
    "code_tensor",
    "fisher_diag_sample",
    "hadamard",
    "residual_probe",
]

from __future__ import annotations
from typing import Literal

from dataclasses import dataclass

import numpy as np

from crossflow_simulator.cross_section import CrossSection


@dataclass
class PillarOperator:
    """
    First surrogate model for a pillar-induced cross-sectional deformation.

    The pillar is represented by a smooth displacement field centered at
    (center_y, center_z). This is not yet a CFD-derived advection map.

    Parameters
    ----------
    center_y, center_z:
        Pillar center in normalized cross-section coordinates.

    radius:
        Effective pillar radius in normalized units.

    strength:
        Overall deformation strength.

    direction:
        +1 and -1 reverse the deformation direction.
    """

    center_y: float = 0.5
    center_z: float = 0.5
    radius: float = 0.25
    strength: float = 0.15
    direction: float = 1.0

    def apply(self, cross_section: CrossSection) -> CrossSection:
        if self.radius <= 0:
            raise ValueError("radius must be positive.")

        result = cross_section.copy()

        Y, Z = np.meshgrid(result.y, result.z, indexing="xy")

        dy = Y - self.center_y
        dz = Z - self.center_z

        r2 = dy**2 + dz**2

        influence = np.exp(-r2 / (2.0 * self.radius**2))

        displacement_y = self.direction * self.strength * dz * influence
        displacement_z = -self.direction * self.strength * dy * influence

        source_y = Y - displacement_y
        source_z = Z - displacement_z

        source_y = np.clip(source_y, 0.0, 1.0)
        source_z = np.clip(source_z, 0.0, 1.0)

        source_j = np.rint(source_y * (result.ny - 1)).astype(int)
        source_i = np.rint(source_z * (result.nz - 1)).astype(int)

        result.labels = cross_section.labels[source_i, source_j]

        return result
    
@dataclass
class PartialHeightPillarOperator:
    """
    Surrogate model for a partial-height pillar.

    Supported shapes:
    - gaussian: smooth influence blob
    - rectangular: block-like pillar footprint
    - triangular: projected triangular footprint inspired by Zou Fig. 3a

    Supported mounts:
    - bottom: pillar grows upward from z = 0
    - top: pillar grows downward from z = 1

    For triangular pillars:
    - the physical footprint lies in the x-y plane
    - the base is aligned along x
    - the apex points along +y or -y
    - the base vertices are (base_x1, base_y1) and (base_x2, base_y1)
    - the third vertex is:
        ((base_x1 + base_x2) / 2,
         base_y1 ± sqrt(3)/2 * (base_x2 - base_x1))
    - the effective y-z influence narrows with height and shifts toward the apex
    """

    center_y: float = 0.5
    height_ratio: float = 0.5
    width_ratio: float = 0.25
    strength_y: float = 0.20
    strength_z: float = 0.10
    direction: float = 1.0
    shape: Literal["gaussian", "rectangular", "triangular"] = "gaussian"
    mount: Literal["bottom", "top"] = "bottom"
    apex_direction: Literal["positive_y", "negative_y"] = "positive_y"

    # Used only for shape="triangular".
    # These describe the triangular footprint in the x-y plane.
    base_x1: float = 0.0
    base_x2: float = 1.0
    base_y1: float = 0.25

    def _triangular_y_bounds(self) -> tuple[float, float, float]:
        """
        Return y_base, y_apex, and triangle_y_span for the x-y triangular footprint.
        """

        if self.base_x2 <= self.base_x1:
            raise ValueError("base_x2 must be greater than base_x1.")

        base_length_x = self.base_x2 - self.base_x1
        triangle_height_y = 0.5 * np.sqrt(3.0) * base_length_x

        y_base = self.base_y1

        if self.apex_direction == "positive_y":
            y_apex = self.base_y1 + triangle_height_y
        elif self.apex_direction == "negative_y":
            y_apex = self.base_y1 - triangle_height_y
        else:
            raise ValueError(
                f"Unknown apex_direction: {self.apex_direction}"
            )

        triangle_y_span = abs(y_apex - y_base)

        return y_base, y_apex, triangle_y_span

    def _shape_mask(self, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        if self.mount == "bottom":
            z_local = Z
            inside_height = (Z >= 0.0) & (Z <= self.height_ratio)
        elif self.mount == "top":
            z_local = 1.0 - Z
            inside_height = (Z <= 1.0) & (Z >= 1.0 - self.height_ratio)
        else:
            raise ValueError(f"Unknown mount: {self.mount}")

        dy = np.abs(Y - self.center_y)

        if self.shape == "gaussian":
            lateral = np.exp(-(dy**2) / (2.0 * self.width_ratio**2))
            vertical = np.exp(
                -((z_local - 0.5 * self.height_ratio) ** 2)
                / (2.0 * (0.5 * self.height_ratio) ** 2)
            )
            return lateral * vertical * inside_height

        if self.shape == "rectangular":
            half_width = 0.5 * self.width_ratio
            return ((dy <= half_width) & inside_height).astype(float)

        if self.shape == "triangular":
            y_base, y_apex, triangle_y_span = self._triangular_y_bounds()

            relative_height = z_local / self.height_ratio
            relative_height = np.clip(relative_height, 0.0, 1.0)

            # At z=0, the projected y-extent is the full y-span of the x-y
            # triangular footprint. At z=height_ratio, it shrinks to zero.
            local_half_width = 0.5 * triangle_y_span * (1.0 - relative_height)

            # The effective influence center shifts from the base y-coordinate
            # toward the apex y-coordinate as height increases.
            effective_center_y = (
                y_base
                + (y_apex - y_base) * relative_height
            )

            dy_shifted = np.abs(Y - effective_center_y)

            mask = (dy_shifted <= local_half_width) & inside_height
            return mask.astype(float)

        raise ValueError(f"Unknown pillar shape: {self.shape}")

    def apply(self, cross_section: CrossSection) -> CrossSection:
        if not 0.0 < self.height_ratio < 1.0:
            raise ValueError("height_ratio must be between 0 and 1.")

        if self.width_ratio <= 0:
            raise ValueError("width_ratio must be positive.")

        result = cross_section.copy()

        Y, Z = np.meshgrid(result.y, result.z, indexing="xy")

        mask = self._shape_mask(Y, Z)

        # Smooth the hard rectangular/triangular masks slightly by mixing
        # with a Gaussian envelope.
        if self.shape == "triangular":
            _, y_apex, triangle_y_span = self._triangular_y_bounds()
            envelope_center_y = 0.5 * (self.base_y1 + y_apex)
            envelope_width = max(triangle_y_span, 1e-12)
            dy_signed = Y - envelope_center_y
            gaussian_envelope = np.exp(
                -(dy_signed**2) / (2.0 * envelope_width**2)
            )
        else:
            dy_signed = Y - self.center_y
            gaussian_envelope = np.exp(
                -(dy_signed**2) / (2.0 * self.width_ratio**2)
            )

        influence = mask * gaussian_envelope

        if self.mount == "bottom":
            vertical_push = self.height_ratio - Z
        else:
            vertical_push = Z - (1.0 - self.height_ratio)

        displacement_y = (
            self.direction
            * self.strength_y
            * dy_signed
            * influence
        )

        displacement_z = (
            self.strength_z
            * vertical_push
            * influence
        )    

        source_y = Y - displacement_y
        source_z = Z - displacement_z

        source_y = np.clip(source_y, 0.0, 1.0)
        source_z = np.clip(source_z, 0.0, 1.0)

        source_j = np.rint(source_y * (result.ny - 1)).astype(int)
        source_i = np.rint(source_z * (result.nz - 1)).astype(int)

        result.labels = cross_section.labels[source_i, source_j]

        return result
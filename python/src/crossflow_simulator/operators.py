#this module defines operators for creating stacked stream layers and array patterns in a crossflow simulation. The PoiseuilleStackingOperator creates vertical or horizontal stacked layers based on flow rates and velocity profiles. The ArrayStackingOperator generates a 2D array/checkered stream pattern. The AutoInletStackingOperator automatically infers the appropriate stacking operator based on inlet coordinates. The InletDiameterDeformationOperator applies a tunable deformation to stacked layers to account for curvature caused by mismatched inlet diameters and channel width.
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from crossflow_simulator.cross_section import CrossSection
from crossflow_simulator.inlets import Inlet, infer_layout_kind, sorted_unique
from crossflow_simulator.physics import boundaries_from_flow_rates


Orientation = Literal["vertical", "horizontal"]


@dataclass
class PoiseuilleStackingOperator:
    """
    Creates vertical or horizontal stacked stream layers.

    The layer boundaries are computed from a velocity profile
    so that each layer carries the requested flow-rate fraction.

    If velocity_profile is None, the default 1D parabolic profile is used
    inside boundaries_from_flow_rates().
    """

    flow_rates: list[float]
    orientation: Orientation = "vertical"
    start_label: int = 1
    velocity_profile: np.ndarray | None = None

    def apply(self, cross_section: CrossSection) -> CrossSection:
        result = cross_section.copy()

        if self.orientation == "vertical":
            boundaries = boundaries_from_flow_rates(
                result.z,
                self.flow_rates,
                velocity=self.velocity_profile,
            )

            for i in range(len(self.flow_rates)):
                result.set_region(
                    y_min=0.0,
                    y_max=1.0,
                    z_min=boundaries[i],
                    z_max=boundaries[i + 1],
                    label=self.start_label + i,
                )

        elif self.orientation == "horizontal":
            boundaries = boundaries_from_flow_rates(
                result.y,
                self.flow_rates,
                velocity=self.velocity_profile,
            )

            for i in range(len(self.flow_rates)):
                result.set_region(
                    y_min=boundaries[i],
                    y_max=boundaries[i + 1],
                    z_min=0.0,
                    z_max=1.0,
                    label=self.start_label + i,
                )

        else:
            raise ValueError(f"Unknown orientation: {self.orientation}")

        return result


@dataclass
class ArrayStackingOperator:
    """
    Creates a 2D array/checkered stream pattern.

    If velocity profiles are provided, row and column boundaries are
    computed using flow-weighted partitioning.
    """

    n_rows: int
    n_cols: int
    start_label: int = 1
    row_flow_rates: list[float] | None = None
    col_flow_rates: list[float] | None = None
    velocity_profile_y: np.ndarray | None = None
    velocity_profile_z: np.ndarray | None = None

    def apply(self, cross_section: CrossSection) -> CrossSection:
        if self.n_rows <= 0 or self.n_cols <= 0:
            raise ValueError("n_rows and n_cols must be positive.")

        result = cross_section.copy()

        row_rates = self.row_flow_rates or [1.0] * self.n_rows
        col_rates = self.col_flow_rates or [1.0] * self.n_cols

        z_boundaries = boundaries_from_flow_rates(
            result.z,
            row_rates,
            velocity=self.velocity_profile_z,
        )

        y_boundaries = boundaries_from_flow_rates(
            result.y,
            col_rates,
            velocity=self.velocity_profile_y,
        )

        label = self.start_label

        for row in range(self.n_rows):
            z_min = z_boundaries[row]
            z_max = z_boundaries[row + 1]

            for col in range(self.n_cols):
                y_min = y_boundaries[col]
                y_max = y_boundaries[col + 1]

                result.set_region(
                    y_min=y_min,
                    y_max=y_max,
                    z_min=z_min,
                    z_max=z_max,
                    label=label,
                )

                label += 1

        return result


@dataclass
class AutoInletStackingOperator:
    """
    Automatically infer and apply the correct stacking operator
    from inlet coordinates.

    Supported layouts:
    - series along x       -> vertical stacking
    - parallel across y    -> horizontal stacking
    - array in y-z plane   -> array stacking
    """

    inlets: list[Inlet]
    velocity_profile_y: np.ndarray | None = None
    velocity_profile_z: np.ndarray | None = None

    def apply(self, cross_section: CrossSection) -> CrossSection:
        layout = infer_layout_kind(self.inlets)

        flow_rates = [inlet.flow_rate for inlet in self.inlets]
        start_label = min(inlet.label for inlet in self.inlets)

        if layout == "vertical":
            operator = PoiseuilleStackingOperator(
                flow_rates=flow_rates,
                orientation="vertical",
                start_label=start_label,
                velocity_profile=self.velocity_profile_z,
            )
            return operator.apply(cross_section)

        if layout == "horizontal":
            operator = PoiseuilleStackingOperator(
                flow_rates=flow_rates,
                orientation="horizontal",
                start_label=start_label,
                velocity_profile=self.velocity_profile_y,
            )
            return operator.apply(cross_section)

        if layout == "array":
            y_values = sorted_unique([inlet.y for inlet in self.inlets])
            z_values = sorted_unique([inlet.z for inlet in self.inlets])

            operator = ArrayStackingOperator(
                n_rows=len(z_values),
                n_cols=len(y_values),
                start_label=start_label,
                velocity_profile_y=self.velocity_profile_y,
                velocity_profile_z=self.velocity_profile_z,
            )
            return operator.apply(cross_section)

        raise ValueError(f"Unsupported inferred layout: {layout}")

@dataclass
class InletDiameterDeformationOperator:
    """
    Apply a tunable inlet-diameter deformation to stacked layers.

    This surrogate captures curvature caused by mismatch between total
    inlet diameter and main channel width.

    diameter_ratio = sum(D_i) / W_c

    Calibration parameters:
    - strength: overall curvature magnitude
    - exponent: controls curvature shape across y
    - gamma: controls nonlinear response to diameter mismatch

    These parameters can be fitted against CFD or experimental
    cross-sectional profiles.
    """

    diameter_ratio: float     # sum(D_i) / W_c 
    strength: float = 0.18    # overall curvature magnitude
    exponent: float = 1.0      # controls curvature shape across y
    gamma: float = 1.0         # controls nonlinear response to diameter mismatch

    def apply(self, cross_section: CrossSection) -> CrossSection:     #takes cross section and eturns deformed copy 
        if self.diameter_ratio <= 0:                                  #safety check 
            raise ValueError("diameter_ratio must be positive.")

        if self.strength < 0:                                         #safety check
            raise ValueError("strength must be non-negative.")

        if self.exponent <= 0:                                        #safety check
            raise ValueError("exponent must be positive.")

        if self.gamma <= 0:                                            #safety check
            raise ValueError("gamma must be positive.")

        result = cross_section.copy()                                 #prevents modification of original cross section

        mismatch = self.diameter_ratio - 1.0                        #crucial, this is why we use 0 in scripts 

        if abs(mismatch) < 1e-12:                                  #mismatch close to zero i.e. diameter ratio almost 1, no deformation needed, return original cross section
            return result

        amplitude = (                                              #max vertical displacement 
            self.strength                                         
            * np.sign(mismatch)                                    
            * abs(mismatch) ** self.gamma
        )  

        """
        #sign of mismatch, 
        #positive if diameter ratio > 1 (inlet diameter sum larger than channel width), 
        #negative if diameter ratio < 1 (inlet diameter sum smaller than channel width)
        #hence amplitude positive i,e upward deformation for diameter ratio > 1, downward deformation for diameter ratio < 1
        """

        y = result.y                                    #normalized lateral coordinates from 0 to 1 across the channel width

        base_shape = 1.0 - 4.0 * (y - 0.5) ** 2         #base parabolic shape centered at y=0.5, with value 1 at center and 0 at edges y=0 and y=1
        base_shape = np.clip(base_shape, 0.0, 1.0)      #clip to ensure non-negative values, shape is zero at edges and positive in the middle, creating a bell-shaped curve across y

        shape = base_shape**self.exponent     

        """
         #adjust shape using exponent, higher exponent creates a more peaked shape, lower exponent creates a flatter shape, exponent=1 keeps the original parabolic shape
        """         

        deformed_labels = np.full_like(result.labels, result.default_label) #initialize deformed labels with default label, will be overwritten by shifted labels

        for j in range(result.ny):         #loop over columns (y direction) to apply vertical shifts to each column of labels based on the deformation shape and amplitude
            shift = amplitude * shape[j]
            shift_pixels = int(round(shift * result.nz))

            deformed_labels[:, j] = np.roll(result.labels[:, j], shift_pixels)

            if shift_pixels > 0:
                deformed_labels[:shift_pixels, j] = result.labels[0, j]
            elif shift_pixels < 0:
                deformed_labels[shift_pixels:, j] = result.labels[-1, j]

        result.labels = deformed_labels

        return result
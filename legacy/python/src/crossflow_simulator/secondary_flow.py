from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from crossflow_simulator.cross_section import CrossSection


VortexDirection = Literal["auto", "clockwise", "counter_clockwise"]


def nearest_label_sample_yz(
    labels: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    y_query: np.ndarray,
    z_query: np.ndarray,
) -> np.ndarray:
    if labels.shape != (len(z), len(y)):
        raise ValueError("labels must have shape (len(z), len(y)).")

    y_query = np.clip(y_query, y[0], y[-1])
    z_query = np.clip(z_query, z[0], z[-1])

    j = np.rint((y_query - y[0]) / (y[-1] - y[0]) * (len(y) - 1)).astype(int)
    i = np.rint((z_query - z[0]) / (z[-1] - z[0]) * (len(z) - 1)).astype(int)

    j = np.clip(j, 0, len(y) - 1)
    i = np.clip(i, 0, len(z) - 1)

    return labels[i, j]


@dataclass
class SlopeInducedVortexParameters:
    center_y: float
    center_z: float
    radius_y: float
    radius_z: float
    circulation: float
    direction_sign: float


def infer_vortex_parameters_from_slope(
    slope_start: tuple[float, float],
    slope_end: tuple[float, float],
    height_ratio: float,
    strength_scale: float = 1.0,
    radius_scale_y: float = 0.50,
    radius_scale_z: float = 0.60,
    center_z_factor: float = 0.55,
    direction: VortexDirection = "auto",
) -> SlopeInducedVortexParameters:
    if not 0.0 < height_ratio < 1.0:
        raise ValueError("height_ratio must be between 0 and 1.")

    x0, y0 = slope_start
    x1, y1 = slope_end

    dx = x1 - x0
    dy = y1 - y0

    slope_length = float(np.sqrt(dx**2 + dy**2))
    if slope_length <= 1e-12:
        raise ValueError("slope_start and slope_end must be distinct.")

    effective_y_span = max(abs(dy), 0.15)

    center_y = 0.5 * (y0 + y1)
    center_z = center_z_factor * height_ratio
    radius_y = max(radius_scale_y * effective_y_span, 1e-3)
    radius_z = max(radius_scale_z * height_ratio, 1e-3)

    if direction == "auto":
        direction_sign = -float(np.sign(dy / dx)) if abs(dx) > 1e-12 else 1.0
        if abs(direction_sign) < 1e-12:
            direction_sign = 1.0
    elif direction == "clockwise":
        direction_sign = -1.0
    elif direction == "counter_clockwise":
        direction_sign = 1.0
    else:
        raise ValueError(
            "direction must be 'auto', 'clockwise', or 'counter_clockwise'."
        )

    circulation = strength_scale * slope_length * height_ratio

    return SlopeInducedVortexParameters(
        center_y=center_y,
        center_z=center_z,
        radius_y=radius_y,
        radius_z=radius_z,
        circulation=circulation,
        direction_sign=direction_sign,
    )


def localized_vortex_velocity_field(
    y: np.ndarray,
    z: np.ndarray,
    center_y: float,
    center_z: float,
    radius_y: float,
    radius_z: float,
    circulation: float,
    direction_sign: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if radius_y <= 0.0:
        raise ValueError("radius_y must be positive.")
    if radius_z <= 0.0:
        raise ValueError("radius_z must be positive.")

    Y, Z = np.meshgrid(y, z, indexing="xy")

    dy = Y - center_y
    dz = Z - center_z

    gaussian = np.exp(-0.5 * ((dy / radius_y) ** 2 + (dz / radius_z) ** 2))
    psi = direction_sign * circulation * gaussian

    velocity_y = psi * (-(dz) / (radius_z**2))
    velocity_z = -psi * (-(dy) / (radius_y**2))

    return velocity_y, velocity_z, psi


@dataclass
class SlopeInducedVortexOperator:
    slope_start: tuple[float, float]
    slope_end: tuple[float, float]
    height_ratio: float

    strength_scale: float = 1.0
    radius_scale_y: float = 0.50
    radius_scale_z: float = 0.60
    center_z_factor: float = 0.55
    direction: VortexDirection = "auto"

    n_steps: int = 80
    normalize_velocity: bool = True
    pseudo_time: float = 1.0

    def infer_parameters(self) -> SlopeInducedVortexParameters:
        return infer_vortex_parameters_from_slope(
            slope_start=self.slope_start,
            slope_end=self.slope_end,
            height_ratio=self.height_ratio,
            strength_scale=self.strength_scale,
            radius_scale_y=self.radius_scale_y,
            radius_scale_z=self.radius_scale_z,
            center_z_factor=self.center_z_factor,
            direction=self.direction,
        )

    def velocity_field(
        self,
        y: np.ndarray,
        z: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        params = self.infer_parameters()

        return localized_vortex_velocity_field(
            y=y,
            z=z,
            center_y=params.center_y,
            center_z=params.center_z,
            radius_y=params.radius_y,
            radius_z=params.radius_z,
            circulation=params.circulation,
            direction_sign=params.direction_sign,
        )

    def apply(self, cross_section: CrossSection) -> CrossSection:
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive.")
        if self.pseudo_time <= 0.0:
            raise ValueError("pseudo_time must be positive.")

        result = cross_section.copy()
        y = result.y
        z = result.z
        Y, Z = np.meshgrid(y, z, indexing="xy")

        velocity_y, velocity_z, _ = self.velocity_field(y=y, z=z)

        if self.normalize_velocity:
            speed = np.sqrt(velocity_y**2 + velocity_z**2)
            max_speed = float(np.max(speed))
            if max_speed > 1e-12:
                velocity_y = velocity_y / max_speed
                velocity_z = velocity_z / max_speed

        labels = result.labels.copy()
        dt = self.pseudo_time / self.n_steps

        for _ in range(self.n_steps):
            source_y = np.clip(Y - velocity_y * dt, 0.0, 1.0)
            source_z = np.clip(Z - velocity_z * dt, 0.0, 1.0)

            labels = nearest_label_sample_yz(
                labels=labels,
                y=y,
                z=z,
                y_query=source_y,
                z_query=source_z,
            )

        result.labels = labels
        return result


@dataclass
class WallBoundedSlopeVortexOperator:
    slope_start: tuple[float, float]
    slope_end: tuple[float, float]
    height_ratio: float

    strength_scale: float = 1.0
    radius_scale_y: float = 0.45
    radius_scale_z: float = 0.45
    center_z_factor: float = 0.55
    direction: VortexDirection = "auto"

    n_steps: int = 100
    pseudo_time: float = 1.0
    normalize_velocity: bool = True

    def infer_parameters(self) -> SlopeInducedVortexParameters:
        return infer_vortex_parameters_from_slope(
            slope_start=self.slope_start,
            slope_end=self.slope_end,
            height_ratio=self.height_ratio,
            strength_scale=self.strength_scale,
            radius_scale_y=self.radius_scale_y,
            radius_scale_z=self.radius_scale_z,
            center_z_factor=self.center_z_factor,
            direction=self.direction,
        )

    def velocity_field(
        self,
        y: np.ndarray,
        z: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        params = self.infer_parameters()

        Y, Z = np.meshgrid(y, z, indexing="xy")

        dy = Y - params.center_y
        dz = Z - params.center_z

        envelope = np.exp(
            -0.5
            * (
                (dy / params.radius_y) ** 2
                + (dz / params.radius_z) ** 2
            )
        )

        wall = np.sin(np.pi * Y) * np.sin(np.pi * Z)

        psi = params.direction_sign * params.circulation * wall * envelope

        dpsi_dz, dpsi_dy = np.gradient(psi, z, y)

        velocity_y = dpsi_dz
        velocity_z = -dpsi_dy

        return velocity_y, velocity_z, psi

    def apply(self, cross_section: CrossSection) -> CrossSection:
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive.")
        if self.pseudo_time <= 0.0:
            raise ValueError("pseudo_time must be positive.")

        result = cross_section.copy()
        y = result.y
        z = result.z
        Y, Z = np.meshgrid(y, z, indexing="xy")

        velocity_y, velocity_z, _ = self.velocity_field(y=y, z=z)

        if self.normalize_velocity:
            speed = np.sqrt(velocity_y**2 + velocity_z**2)
            max_speed = float(np.max(speed))
            if max_speed > 1e-12:
                velocity_y = velocity_y / max_speed
                velocity_z = velocity_z / max_speed

        labels = result.labels.copy()
        dt = self.pseudo_time / self.n_steps

        for _ in range(self.n_steps):
            source_y = np.clip(Y - velocity_y * dt, 0.0, 1.0)
            source_z = np.clip(Z - velocity_z * dt, 0.0, 1.0)

            labels = nearest_label_sample_yz(
                labels=labels,
                y=y,
                z=z,
                y_query=source_y,
                z_query=source_z,
            )

        result.labels = labels
        return result


@dataclass
class SlopeDislocationVortexOperator:
    slope_start: tuple[float, float]
    slope_end: tuple[float, float]
    height_ratio: float

    dislocation_strength: float = 0.12
    vortex_strength_scale: float = 0.8
    radius_scale_y: float = 0.35
    radius_scale_z: float = 0.25
    center_z_factor: float = 0.45
    direction: VortexDirection = "auto"

    vortex_steps: int = 100
    vortex_pseudo_time: float = 0.15
    normalize_velocity: bool = True

    def infer_parameters(self) -> SlopeInducedVortexParameters:
        return infer_vortex_parameters_from_slope(
            slope_start=self.slope_start,
            slope_end=self.slope_end,
            height_ratio=self.height_ratio,
            strength_scale=self.vortex_strength_scale,
            radius_scale_y=self.radius_scale_y,
            radius_scale_z=self.radius_scale_z,
            center_z_factor=self.center_z_factor,
            direction=self.direction,
        )

    def _apply_dislocation(
        self,
        labels: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> np.ndarray:
        params = self.infer_parameters()

        Y, Z = np.meshgrid(y, z, indexing="xy")

        dy = Y - params.center_y
        dz = Z - params.center_z

        envelope = np.exp(
            -0.5
            * (
                (dy / params.radius_y) ** 2
                + (dz / params.radius_z) ** 2
            )
        )

        partial_height_mask = (Z <= self.height_ratio).astype(float)

        displacement_y = (
            params.direction_sign
            * self.dislocation_strength
            * envelope
            * partial_height_mask
        )

        source_y = np.clip(Y - displacement_y, 0.0, 1.0)
        source_z = Z

        return nearest_label_sample_yz(
            labels=labels,
            y=y,
            z=z,
            y_query=source_y,
            z_query=source_z,
        )

    def apply(self, cross_section: CrossSection) -> CrossSection:
        result = cross_section.copy()

        labels_after_dislocation = self._apply_dislocation(
            labels=result.labels,
            y=result.y,
            z=result.z,
        )

        intermediate = result.copy()
        intermediate.labels = labels_after_dislocation

        vortex = WallBoundedSlopeVortexOperator(
            slope_start=self.slope_start,
            slope_end=self.slope_end,
            height_ratio=self.height_ratio,
            strength_scale=self.vortex_strength_scale,
            radius_scale_y=self.radius_scale_y,
            radius_scale_z=self.radius_scale_z,
            center_z_factor=self.center_z_factor,
            direction=self.direction,
            n_steps=self.vortex_steps,
            pseudo_time=self.vortex_pseudo_time,
            normalize_velocity=self.normalize_velocity,
        )

        return vortex.apply(intermediate)


@dataclass
class VerticalDislocationOperator:
    center_y: float = 0.5
    center_z: float = 0.35
    radius_y: float = 0.25
    radius_z: float = 0.25
    strength_z: float = 0.10
    direction_sign: float = 1.0
    restrict_to_partial_height: bool = True
    height_ratio: float = 0.67

    def apply(self, cross_section: CrossSection) -> CrossSection:
        result = cross_section.copy()
        y = result.y
        z = result.z

        Y, Z = np.meshgrid(y, z, indexing="xy")

        dy = Y - self.center_y
        dz = Z - self.center_z

        envelope = np.exp(
            -0.5
            * (
                (dy / self.radius_y) ** 2
                + (dz / self.radius_z) ** 2
            )
        )

        if self.restrict_to_partial_height:
            envelope = envelope * (Z <= self.height_ratio).astype(float)

        displacement_z = self.direction_sign * self.strength_z * envelope

        source_y = Y
        source_z = np.clip(Z - displacement_z, 0.0, 1.0)

        result.labels = nearest_label_sample_yz(
            labels=result.labels,
            y=y,
            z=z,
            y_query=source_y,
            z_query=source_z,
        )

        return result


@dataclass
class PairedSlopeSecondaryFlowOperator:
    strength: float = 0.20
    direction_sign: float = 1.0
    n_steps: int = 100
    pseudo_time: float = 0.20
    normalize_velocity: bool = True

    def velocity_field(
        self,
        y: np.ndarray,
        z: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        Y, Z = np.meshgrid(y, z, indexing="xy")

        psi = (
            self.direction_sign
            * self.strength
            * np.sin(2.0 * np.pi * Y)
            * np.sin(np.pi * Z)
        )

        dpsi_dz, dpsi_dy = np.gradient(psi, z, y)

        velocity_y = dpsi_dz
        velocity_z = -dpsi_dy

        return velocity_y, velocity_z, psi

    def apply(self, cross_section: CrossSection) -> CrossSection:
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive.")
        if self.pseudo_time <= 0.0:
            raise ValueError("pseudo_time must be positive.")

        result = cross_section.copy()
        y = result.y
        z = result.z
        Y, Z = np.meshgrid(y, z, indexing="xy")

        velocity_y, velocity_z, _ = self.velocity_field(y=y, z=z)

        if self.normalize_velocity:
            speed = np.sqrt(velocity_y**2 + velocity_z**2)
            max_speed = float(np.max(speed))
            if max_speed > 1e-12:
                velocity_y = velocity_y / max_speed
                velocity_z = velocity_z / max_speed

        labels = result.labels.copy()
        dt = self.pseudo_time / self.n_steps

        for _ in range(self.n_steps):
            source_y = np.clip(Y - velocity_y * dt, 0.0, 1.0)
            source_z = np.clip(Z - velocity_z * dt, 0.0, 1.0)

            labels = nearest_label_sample_yz(
                labels=labels,
                y=y,
                z=z,
                y_query=source_y,
                z_query=source_z,
            )

        result.labels = labels
        return result


def slope_metrics(
    slope_start: tuple[float, float],
    slope_end: tuple[float, float],
) -> dict[str, float]:
    x0, y0 = slope_start
    x1, y1 = slope_end

    dx = x1 - x0
    dy = y1 - y0

    length = float(np.sqrt(dx**2 + dy**2))
    if length <= 1e-12:
        raise ValueError("Slope endpoints must be distinct.")

    direction_sign = -float(np.sign(dy / dx)) if abs(dx) > 1e-12 else 1.0

    return {
        "dx": dx,
        "dy": dy,
        "length": length,
        "obliqueness": abs(dy) / length,
        "streamwise_projection": abs(dx) / length,
        "direction_sign": direction_sign,
    }


def geometry_derived_slope_strengths(
    slope_start: tuple[float, float],
    slope_end: tuple[float, float],
    height_ratio: float,
    c_dislocation: float = 0.25,
    c_vortex: float = 1.00,
    c_vertical: float = 0.15,
) -> dict[str, float]:
    if not 0.0 < height_ratio < 1.0:
        raise ValueError("height_ratio must be between 0 and 1.")

    metrics = slope_metrics(
        slope_start=slope_start,
        slope_end=slope_end,
    )

    dislocation_strength = c_dislocation * height_ratio * metrics["length"]
    vortex_strength_scale = c_vortex * height_ratio * metrics["obliqueness"]
    vertical_strength = c_vertical * height_ratio * metrics["streamwise_projection"]

    return {
        "dislocation_strength": float(dislocation_strength),
        "vortex_strength_scale": float(vortex_strength_scale),
        "vertical_strength": float(vertical_strength),
        "direction_sign": float(metrics["direction_sign"]),
        "slope_length": float(metrics["length"]),
        "obliqueness": float(metrics["obliqueness"]),
        "streamwise_projection": float(metrics["streamwise_projection"]),
    }


def inplane_divergence(
    x: np.ndarray,
    y: np.ndarray,
    velocity_x: np.ndarray,
    velocity_y: np.ndarray,
) -> np.ndarray:
    if velocity_x.shape != (len(y), len(x)):
        raise ValueError("velocity_x must have shape (len(y), len(x)).")
    if velocity_y.shape != (len(y), len(x)):
        raise ValueError("velocity_y must have shape (len(y), len(x)).")

    dux_dx = np.gradient(velocity_x, x, axis=1)
    duy_dy = np.gradient(velocity_y, y, axis=0)

    return dux_dx + duy_dy


def vertical_forcing_from_lubrication(
    x: np.ndarray,
    y: np.ndarray,
    velocity_x: np.ndarray,
    velocity_y: np.ndarray,
    pillar_height: np.ndarray | None = None,
    smooth: bool = True,
) -> np.ndarray:
    divergence = inplane_divergence(
        x=x,
        y=y,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
    )

    vertical_source = -divergence

    if pillar_height is not None:
        if pillar_height.shape != vertical_source.shape:
            raise ValueError("pillar_height must match velocity field shape.")

        max_height = max(float(np.max(pillar_height)), 1e-12)
        weight = pillar_height / max_height
        vertical_source = vertical_source * weight

    forcing_y = np.trapezoid(vertical_source, x, axis=1)

    forcing_y = forcing_y - float(np.mean(forcing_y))

    if smooth:
        kernel = np.array(
            [1.0, 2.0, 4.0, 8.0, 12.0, 8.0, 4.0, 2.0, 1.0],
            dtype=float,
        )
        kernel = kernel / np.sum(kernel)

        # Apply twice to suppress polygon-edge oscillations.
        forcing_y = np.convolve(forcing_y, kernel, mode="same")
        forcing_y = np.convolve(forcing_y, kernel, mode="same")

    max_abs = float(np.max(np.abs(forcing_y)))

    if max_abs > 1e-12:
        forcing_y = forcing_y / max_abs

    return forcing_y


@dataclass
class DivergenceDrivenVerticalDislocationOperator:
    forcing_y: np.ndarray
    strength: float = 0.12

    def apply(self, cross_section: CrossSection) -> CrossSection:
        result = cross_section.copy()
        y = result.y
        z = result.z

        if self.forcing_y.shape != y.shape:
            raise ValueError("forcing_y must have shape matching cross_section.y.")

        Y, Z = np.meshgrid(y, z, indexing="xy")

        forcing_interp = np.interp(Y, y, self.forcing_y)
        wall_shape = np.sin(np.pi * Z)

        displacement_z = self.strength * forcing_interp * wall_shape

        source_y = Y
        source_z = np.clip(Z - displacement_z, 0.0, 1.0)

        result.labels = nearest_label_sample_yz(
            labels=result.labels,
            y=y,
            z=z,
            y_query=source_y,
            z_query=source_z,
        )

        return result
    
@dataclass
class DivergenceDrivenSecondaryFlowOperator:
    """
    Secondary-flow operator driven by lubrication-field divergence.

    This converts the y-dependent forcing obtained from -div(u_x, u_y)
    into a wall-bounded streamfunction in the y-z cross-section:

        psi(y,z) = strength * A(y) * sin(pi z)

    where A(y) is the normalized divergence-derived forcing profile.

    The induced velocity field is incompressible in the y-z plane:

        v_y = d psi / d z
        v_z = - d psi / d y

    This is intended to produce coherent secondary circulation rather than
    direct vertical label displacement.
    """

    forcing_y: np.ndarray
    strength: float = 0.08
    n_steps: int = 100
    pseudo_time: float = 0.10
    normalize_velocity: bool = True

    def velocity_field(
        self,
        y: np.ndarray,
        z: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.forcing_y.shape != y.shape:
            raise ValueError("forcing_y must have shape matching y.")

        Y, Z = np.meshgrid(y, z, indexing="xy")

        forcing_interp = np.interp(
            Y,
            y,
            self.forcing_y,
        )

        psi = (
            self.strength
            * forcing_interp
            * np.sin(np.pi * Z)
        )

        dpsi_dz, dpsi_dy = np.gradient(psi, z, y)

        velocity_y = dpsi_dz
        velocity_z = -dpsi_dy

        return velocity_y, velocity_z, psi

    def apply(self, cross_section: CrossSection) -> CrossSection:
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive.")

        if self.pseudo_time <= 0.0:
            raise ValueError("pseudo_time must be positive.")

        result = cross_section.copy()

        y = result.y
        z = result.z

        Y, Z = np.meshgrid(y, z, indexing="xy")

        velocity_y, velocity_z, _ = self.velocity_field(
            y=y,
            z=z,
        )

        if self.normalize_velocity:
            speed = np.sqrt(velocity_y**2 + velocity_z**2)
            max_speed = float(np.max(speed))

            if max_speed > 1e-12:
                velocity_y = velocity_y / max_speed
                velocity_z = velocity_z / max_speed

        labels = result.labels.copy()
        dt = self.pseudo_time / self.n_steps

        for _ in range(self.n_steps):
            source_y = Y - velocity_y * dt
            source_z = Z - velocity_z * dt

            source_y = np.clip(source_y, 0.0, 1.0)
            source_z = np.clip(source_z, 0.0, 1.0)

            labels = nearest_label_sample_yz(
                labels=labels,
                y=y,
                z=z,
                y_query=source_y,
                z_query=source_z,
            )

        result.labels = labels

        return result
    
@dataclass
class DivergenceDrivenLocalizedSecondaryFlowOperator:
    """
    Localized secondary-flow operator driven by lubrication-field divergence.

    This improves on DivergenceDrivenSecondaryFlowOperator by localizing the
    streamfunction around the pillar-affected region:

        psi(y,z) = strength * A(y) * wall(y,z) * envelope(y,z)

    where:
        A(y) comes from integrated -div(u_x, u_y)
        wall(y,z) = sin(pi y) sin(pi z)
        envelope localizes the circulation near the active pillar region

    The velocity field is incompressible in the y-z plane:

        v_y = d psi / d z
        v_z = - d psi / d y
    """

    forcing_y: np.ndarray
    center_y: float = 0.5
    center_z: float = 0.45
    radius_y: float = 0.30
    radius_z: float = 0.30
    strength: float = 0.08
    n_steps: int = 120
    pseudo_time: float = 0.12
    normalize_velocity: bool = True

    def velocity_field(
        self,
        y: np.ndarray,
        z: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.forcing_y.shape != y.shape:
            raise ValueError("forcing_y must have shape matching y.")

        if self.radius_y <= 0.0:
            raise ValueError("radius_y must be positive.")

        if self.radius_z <= 0.0:
            raise ValueError("radius_z must be positive.")

        Y, Z = np.meshgrid(y, z, indexing="xy")

        forcing_interp = np.interp(
            Y,
            y,
            self.forcing_y,
        )

        dy = Y - self.center_y
        dz = Z - self.center_z

        envelope = np.exp(
            -0.5
            * (
                (dy / self.radius_y) ** 2
                + (dz / self.radius_z) ** 2
            )
        )

        wall = np.sin(np.pi * Y) * np.sin(np.pi * Z)

        psi = (
            self.strength
            * forcing_interp
            * wall
            * envelope
        )

        dpsi_dz, dpsi_dy = np.gradient(psi, z, y)

        velocity_y = dpsi_dz
        velocity_z = -dpsi_dy

        return velocity_y, velocity_z, psi

    def apply(self, cross_section: CrossSection) -> CrossSection:
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive.")

        if self.pseudo_time <= 0.0:
            raise ValueError("pseudo_time must be positive.")

        result = cross_section.copy()

        y = result.y
        z = result.z

        Y, Z = np.meshgrid(y, z, indexing="xy")

        velocity_y, velocity_z, _ = self.velocity_field(
            y=y,
            z=z,
        )

        if self.normalize_velocity:
            speed = np.sqrt(velocity_y**2 + velocity_z**2)
            max_speed = float(np.max(speed))

            if max_speed > 1e-12:
                velocity_y = velocity_y / max_speed
                velocity_z = velocity_z / max_speed

        labels = result.labels.copy()
        dt = self.pseudo_time / self.n_steps

        for _ in range(self.n_steps):
            source_y = Y - velocity_y * dt
            source_z = Z - velocity_z * dt

            source_y = np.clip(source_y, 0.0, 1.0)
            source_z = np.clip(source_z, 0.0, 1.0)

            labels = nearest_label_sample_yz(
                labels=labels,
                y=y,
                z=z,
                y_query=source_y,
                z_query=source_z,
            )

        result.labels = labels

        return result
    

def local_vertical_forcing_from_lubrication(
    x: np.ndarray,
    y: np.ndarray,
    velocity_x: np.ndarray,
    velocity_y: np.ndarray,
    slope_start: tuple[float, float],
    slope_end: tuple[float, float],
    pillar_height: np.ndarray | None = None,
    window_scale: float = 0.20,
    smooth: bool = True,
) -> np.ndarray:
    divergence = inplane_divergence(
        x=x,
        y=y,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
    )

    vertical_source = -divergence

    if pillar_height is not None:
        if pillar_height.shape != vertical_source.shape:
            raise ValueError("pillar_height must match velocity field shape.")

        max_height = max(float(np.max(pillar_height)), 1e-12)
        vertical_source = vertical_source * (pillar_height / max_height)

    x0, _ = slope_start
    x1, _ = slope_end
    x_center = 0.5 * (x0 + x1)

    slope_length = float(
        np.sqrt(
            (slope_end[0] - slope_start[0]) ** 2
            + (slope_end[1] - slope_start[1]) ** 2
        )
    )

    sigma_x = max(window_scale * slope_length, 1e-3)

    x_weight = np.exp(-0.5 * ((x - x_center) / sigma_x) ** 2)
    x_weight = x_weight / max(float(np.max(x_weight)), 1e-12)

    local_source = vertical_source * x_weight[None, :]

    forcing_y = np.trapezoid(local_source, x, axis=1)
    forcing_y = forcing_y - float(np.mean(forcing_y))

    if smooth:
        kernel = np.array(
            [1.0, 2.0, 4.0, 8.0, 12.0, 8.0, 4.0, 2.0, 1.0],
            dtype=float,
        )
        kernel = kernel / np.sum(kernel)

        forcing_y = np.convolve(forcing_y, kernel, mode="same")
        forcing_y = np.convolve(forcing_y, kernel, mode="same")

    max_abs = float(np.max(np.abs(forcing_y)))

    if max_abs > 1e-12:
        forcing_y = forcing_y / max_abs

    return forcing_y


def slope_center_from_endpoints(
    slope_start: tuple[float, float],
    slope_end: tuple[float, float],
    center_z_factor: float,
    height_ratio: float,
) -> tuple[float, float]:
    _, y0 = slope_start
    _, y1 = slope_end

    center_y = 0.5 * (y0 + y1)
    center_z = center_z_factor * height_ratio

    return float(center_y), float(center_z)

def low_mode_filter_forcing_y(y, forcing_y, n_modes=2):
    filtered = np.zeros_like(forcing_y)

    for k in range(1, n_modes + 1):
        basis = np.sin(k * np.pi * y)
        coeff = np.trapezoid(forcing_y * basis, y) / np.trapezoid(basis * basis, y)
        filtered += coeff * basis

    max_abs = np.max(np.abs(filtered))
    if max_abs > 1e-12:
        filtered /= max_abs

    return filtered
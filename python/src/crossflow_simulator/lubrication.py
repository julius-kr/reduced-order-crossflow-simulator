#this module implements the lubrication approximation for flow in a rectangular channel with bottom-mounted pillars. It provides functions to compute pillar height maps, gap heights, hydraulic conductance, solve the lubrication pressure field, compute depth-averaged velocities, and advect labels through the velocity field. The code supports both triangular and generic polygonal pillar footprints and allows for visualization of outlet cross-sections.
from __future__ import annotations
from typing import Sequence

from dataclasses import dataclass

import numpy as np


@dataclass
class TriangularPillarFootprint:
    """
    Triangular pillar footprint in the x-y plane.

    The base lies along the x direction:
        A = (x1, y1)
        B = (x2, y1)

    The apex points in +y or -y:
        C = ((x1 + x2)/2, y1 +/- sqrt(3)/2 * (x2 - x1))
    """

    x1: float     # Base start x-coordinate.
    x2: float     # Base end x-coordinate.
    y1: float     # Base y-coordinate.
    height_ratio: float  # Height ratio relative to channel height.
    apex_direction: str = "positive_y"  # Direction of the apex.

    def vertices(self) -> np.ndarray:       # Return vertices as a (3, 2) array of (x, y) pairs.
        if self.x2 <= self.x1:
            raise ValueError("x2 must be greater than x1.")

        if not 0.0 < self.height_ratio < 1.0:
            raise ValueError("height_ratio must be between 0 and 1.")

        base_length = self.x2 - self.x1
        triangle_height = 0.5 * np.sqrt(3.0) * base_length

        x3 = 0.5 * (self.x1 + self.x2)

        if self.apex_direction == "positive_y":
            y3 = self.y1 + triangle_height
        elif self.apex_direction == "negative_y":
            y3 = self.y1 - triangle_height
        else:
            raise ValueError(
                "apex_direction must be 'positive_y' or 'negative_y'."
            )

        return np.array(
            [
                [self.x1, self.y1],
                [self.x2, self.y1],
                [x3, y3],
            ],
            dtype=float,
        )


def points_in_triangle(
    X: np.ndarray,
    Y: np.ndarray,
    vertices: np.ndarray,
) -> np.ndarray:
    """
    Return a Boolean mask for points inside a triangle.

    Uses barycentric coordinates.
    """

    x = X
    y = Y

    x1, y1 = vertices[0]
    x2, y2 = vertices[1]
    x3, y3 = vertices[2]

    denominator = (
        (y2 - y3) * (x1 - x3)
        + (x3 - x2) * (y1 - y3)
    )

    if abs(denominator) < 1e-15:
        raise ValueError("Triangle vertices are degenerate.")

    a = (
        (y2 - y3) * (x - x3)
        + (x3 - x2) * (y - y3)
    ) / denominator

    b = (
        (y3 - y1) * (x - x3)
        + (x1 - x3) * (y - y3)
    ) / denominator

    c = 1.0 - a - b

    return (a >= 0.0) & (b >= 0.0) & (c >= 0.0)


def triangular_pillar_height_map(     #builds a 2D map of pillar height over the top-view xy domain  
    x: np.ndarray,
    y: np.ndarray,
    footprint: TriangularPillarFootprint,
) -> np.ndarray:
    """
    Build a normalized pillar height map H_p(x, y) / H_c.

    Inside the triangular footprint:
        height = footprint.height_ratio

    Outside:
        height = 0

    Returns array with shape (len(y), len(x)).
    """

    X, Y = np.meshgrid(x, y, indexing="xy")

    vertices = footprint.vertices()
    mask = points_in_triangle(X, Y, vertices)

    height_map = np.zeros_like(X, dtype=float)
    height_map[mask] = footprint.height_ratio

    return height_map


def gap_height_from_pillar(                #converts pillar height to available fluid gap height above the pillar, normalized by channel height
    pillar_height_map: np.ndarray,
) -> np.ndarray:
    """
    Compute normalized gap height h/H_c above a bottom-mounted pillar.

    h/H_c = 1 - H_p/H_c
    """

    if np.any(pillar_height_map < 0.0):
        raise ValueError("pillar_height_map must be non-negative.")

    if np.any(pillar_height_map >= 1.0):
        raise ValueError("pillar height must be smaller than channel height.")

    return 1.0 - pillar_height_map


def conductance_from_gap(                  #converts gap height to hydraulic conductance for lubrication flow, which is proportional to h^3
    gap_height: np.ndarray,
) -> np.ndarray:
    """
    Lubrication conductance proportional to h^3.

    For depth-averaged pressure-driven flow:
        q = -h^3 grad(p)
    up to a constant factor.
    """

    if np.any(gap_height <= 0.0):
        raise ValueError("gap_height must be positive.")

    return gap_height**3

def solve_lubrication_pressure(          #solves \Nabla \cdot (k \Nabla p) = 0 for the pressure field p(x, y) given the conductance field k(x, y); current bottleneck   
    conductance: np.ndarray,
    p_left: float = 1.0,
    p_right: float = 0.0,
    max_iterations: int = 20_000,
    tolerance: float = 1e-8,
    omega: float = 1.6,
) -> np.ndarray:
    """
    Solve div(k grad p) = 0 on a rectangular grid.

    conductance:
        k(x, y) = h(x, y)^3

    Boundary conditions:
    - left boundary:  p = p_left
    - right boundary: p = p_right
    - bottom/top:     no-flux dp/dy = 0

    Uses Gauss-Seidel SOR iteration.
    """

    if conductance.ndim != 2:                            #pressure solver expects a a2D grid of conductance values, one for each (x, y) point
        raise ValueError("conductance must be a 2D array.")

    if np.any(conductance <= 0.0):                       #conductance must be positive everywhere
        raise ValueError("conductance must be positive everywhere.")

    ny, nx = conductance.shape                       #gets grid dimensions

    # Initial pressure guess: linear drop from left to right.
    x = np.linspace(0.0, 1.0, nx)
    p = p_left + (p_right - p_left) * x
    p = np.tile(p, (ny, 1))

    # Dirichlet boundaries.
    p[:, 0] = p_left                 #sets left boundary to p_left#applies fixed pressure at inlet and outlet 
    p[:, -1] = p_right

    for iteration in range(max_iterations):             #iteratively update pressure values until convergence, max_change tracks convergence  
        max_change = 0.0

        # No-flux top/bottom implemented by copying neighboring rows.
        p[0, :] = p[1, :]
        p[-1, :] = p[-2, :]

        # Keep pressure fixed at inlet/outlet.
        p[:, 0] = p_left
        p[:, -1] = p_right

        for j in range(1, ny - 1):                               #loops over every grid point except boundaries 
            for i in range(1, nx - 1):
                k_e = 0.5 * (conductance[j, i] + conductance[j, i + 1])     #effective conductance to the east neighbor, averaged between current and east grid point
                k_w = 0.5 * (conductance[j, i] + conductance[j, i - 1])     #effective conductance to the west neighbor
                k_n = 0.5 * (conductance[j, i] + conductance[j + 1, i])     #effective conductance to the north neighbor
                k_s = 0.5 * (conductance[j, i] + conductance[j - 1, i])     #effective conductance to the south neighbor

                denominator = k_e + k_w + k_n + k_s             #sum of conductances to all four neighbors, used for normalization  

                p_new = (
                    k_e * p[j, i + 1]
                    + k_w * p[j, i - 1]
                    + k_n * p[j + 1, i]
                    + k_s * p[j - 1, i]
                ) / denominator                                 #weighted average of neighboring pressures, weighted by conductance to each neighbor

                updated = (1.0 - omega) * p[j, i] + omega * p_new     #SOR update with relaxation factor 

                """omega=1: standard Gauss-Seidel update, fully replaces old value with new weighted average
                   omega<1: under-relaxation, new value is a blend of old and new
                   omega>1: over-relaxation, new value is extrapolated beyond the new weighted average, can accelerate convergence for certain problems but may cause divergence if too large"""

                change = abs(updated - p[j, i])
                max_change = max(max_change, change)

                p[j, i] = updated      #update pressure at current grid point

        if max_change < tolerance:
            print(f"Pressure solver converged in {iteration + 1} iterations.")      #stops when updates become small, indicating convergence to a solution of the PDE
            break
    else:
        print(
            "Warning: pressure solver did not converge. "
            f"Final max change = {max_change:.3e}"
        )

    return p                            #returns the computed pressure field after convergence or reaching max iterations


def velocity_from_pressure(
    pressure: np.ndarray,
    conductance: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute depth-averaged velocity from pressure.

        u = -k dp/dx
        v = -k dp/dy

    Returns:
        velocity_x, velocity_y
    """

    if pressure.shape != conductance.shape:
        raise ValueError("pressure and conductance must have the same shape.")

    dp_dy, dp_dx = np.gradient(pressure, y, x)

    velocity_x = -conductance * dp_dx
    velocity_y = -conductance * dp_dy

    return velocity_x, velocity_y

def advect_labels_to_outlet(        #first advection attempt, traces particle from left boundary x=0 to outlet x=1  
    inlet_labels: np.ndarray,
    velocity_x: np.ndarray,
    velocity_y: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    n_steps: int = 400,
) -> np.ndarray:
    """
    Advect inlet labels through a 2D depth-averaged velocity field.

    This performs forward particle tracing.

    Each grid point at the inlet x = 0 is treated as a tracer.
    The tracer is moved through the velocity field until it reaches
    the outlet x = 1. Its original label is then deposited at the
    outlet y-location.

    Returns
    -------
    outlet_labels:
        1D array of labels along the outlet boundary, indexed by y.
    """

    if inlet_labels.shape != velocity_x.shape:     #checks if grid shapes match, since we need to sample velocity at the same grid points as the labels for accurate advection
        raise ValueError("inlet_labels and velocity_x must have the same shape.")

    if velocity_x.shape != velocity_y.shape:
        raise ValueError("velocity_x and velocity_y must have the same shape.")

    ny, nx = inlet_labels.shape              #gets grid dimensions

    outlet_labels = np.zeros(ny, dtype=int)   #initializes an array to store the labels at the outlet, one for each y-grid point

    dx = 1.0 / n_steps                       #fixed step size in x direction for particle advection 

    # Start one particle at each inlet y-grid point and stores its starting label 
    for j0 in range(ny):
        particle_x = 0.0
        particle_y = y[j0]

        label = inlet_labels[j0, 0]

        for _ in range(n_steps):
            i = int(np.clip(round(particle_x * (nx - 1)), 0, nx - 1))
            j = int(np.clip(round(particle_y * (ny - 1)), 0, ny - 1))

            ux = velocity_x[j, i]      #gets local velocity 
            uy = velocity_y[j, i]

            if abs(ux) < 1e-12:        #avoids division by zero  
                break

            # March forward by a fixed dx and use dy/dx = uy/ux.
            particle_y += (uy / ux) * dx
            particle_x += dx

            particle_y = float(np.clip(particle_y, 0.0, 1.0))  #keep particle within y bounds of the channel

            if particle_x >= 1.0:
                break

        j_out = int(np.clip(round(particle_y * (ny - 1)), 0, ny - 1)) 
        outlet_labels[j_out] = label                           #deposits the original label of the particle at the outlet y-location corresponding to where it exits the channel at x=1

    # Fill any empty outlet pixels by nearest non-empty label.
    nonzero = np.where(outlet_labels != 0)[0]

    if len(nonzero) == 0:
        raise ValueError("No particles reached outlet.")

    for j in range(ny):
        if outlet_labels[j] == 0:
            nearest = nonzero[np.argmin(np.abs(nonzero - j))]
            outlet_labels[j] = outlet_labels[nearest]

    return outlet_labels


def outlet_labels_to_cross_section(
    outlet_labels: np.ndarray,
    nx_display: int = 80,
) -> np.ndarray:
    """
    Convert 1D outlet labels into a 2D image for visualization.

    The returned array has shape (ny, nx_display).
    """

    return np.tile(outlet_labels[:, None], (1, nx_display))

def bilinear_sample(
    field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    x_query: np.ndarray,
    y_query: np.ndarray,
) -> np.ndarray:
    """
    Bilinearly sample a 2D field at query points.

    field has shape (len(y), len(x)).
    """

    ny, nx = field.shape

    x_query = np.clip(x_query, x[0], x[-1])
    y_query = np.clip(y_query, y[0], y[-1])

    ix = (x_query - x[0]) / (x[-1] - x[0]) * (nx - 1)
    iy = (y_query - y[0]) / (y[-1] - y[0]) * (ny - 1)

    i0 = np.floor(ix).astype(int)
    j0 = np.floor(iy).astype(int)

    i1 = np.clip(i0 + 1, 0, nx - 1)
    j1 = np.clip(j0 + 1, 0, ny - 1)

    wx = ix - i0
    wy = iy - j0

    f00 = field[j0, i0]
    f10 = field[j0, i1]
    f01 = field[j1, i0]
    f11 = field[j1, i1]

    return (
        (1.0 - wx) * (1.0 - wy) * f00
        + wx * (1.0 - wy) * f10
        + (1.0 - wx) * wy * f01
        + wx * wy * f11
    )


def nearest_label_sample(
    labels: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    x_query: np.ndarray,
    y_query: np.ndarray,
) -> np.ndarray:
    """
    Nearest-neighbor sampling for integer label fields.
    """

    ny, nx = labels.shape

    x_query = np.clip(x_query, x[0], x[-1])
    y_query = np.clip(y_query, y[0], y[-1])

    i = np.rint((x_query - x[0]) / (x[-1] - x[0]) * (nx - 1)).astype(int)
    j = np.rint((y_query - y[0]) / (y[-1] - y[0]) * (ny - 1)).astype(int)

    i = np.clip(i, 0, nx - 1)
    j = np.clip(j, 0, ny - 1)

    return labels[j, i]


def advect_label_field_semi_lagrangian(
    labels: np.ndarray,
    velocity_x: np.ndarray,
    velocity_y: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    pseudo_time: float = 0.20,
    n_steps: int = 100,
    normalize_velocity: bool = True,
) -> np.ndarray:
    """
    Advect a full 2D label field using semi-Lagrangian backtracing.

    This treats the label field as a passive scalar transported by the
    depth-averaged lubrication velocity field.

    This is not yet the full 3D y-z outlet cross-section map, but it gives
    a full 2D deformation of the inlet label field.
    """

    if labels.shape != velocity_x.shape:
        raise ValueError("labels and velocity_x must have the same shape.")

    if velocity_x.shape != velocity_y.shape:
        raise ValueError("velocity_x and velocity_y must have the same shape.")

    X, Y = np.meshgrid(x, y, indexing="xy")

    source_x = X.copy()
    source_y = Y.copy()

    vx = velocity_x.copy()
    vy = velocity_y.copy()

    if normalize_velocity:
        speed = np.sqrt(vx**2 + vy**2)
        max_speed = np.max(speed)

        if max_speed <= 0.0:
            raise ValueError("velocity field cannot be zero everywhere.")

        vx = vx / max_speed
        vy = vy / max_speed

    dt = pseudo_time / n_steps

    for _ in range(n_steps):
        sampled_vx = bilinear_sample(vx, x, y, source_x, source_y)
        sampled_vy = bilinear_sample(vy, x, y, source_x, source_y)

        source_x -= sampled_vx * dt
        source_y -= sampled_vy * dt

        source_x = np.clip(source_x, x[0], x[-1])
        source_y = np.clip(source_y, y[0], y[-1])

    return nearest_label_sample(labels, x, y, source_x, source_y)

@dataclass
class ChannelGeometry2D:
    """
    Rectangular channel footprint in the x-y plane.

    x is the flow direction.
    y is the lateral channel direction.
    """

    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0

    def validate(self) -> None:
        if self.x_max <= self.x_min:
            raise ValueError("x_max must be greater than x_min.")

        if self.y_max <= self.y_min:
            raise ValueError("y_max must be greater than y_min.")


@dataclass
class PolygonPillarFootprint:
    """
    Generic polygonal pillar footprint in the x-y plane.

    vertices:
        Polygon vertices as [(x0, y0), (x1, y1), ...].

    height_ratio:
        Pillar height divided by channel height, H_p / H_c.

    This supports asymmetric pillars such as right triangles.
    """

    vertices: Sequence[tuple[float, float]]
    height_ratio: float

    def validate(self) -> None:
        if len(self.vertices) < 3:
            raise ValueError("A polygon footprint needs at least 3 vertices.")

        if not 0.0 < self.height_ratio < 1.0:
            raise ValueError("height_ratio must be between 0 and 1.")

        vertices_array = np.asarray(self.vertices, dtype=float)

        if vertices_array.ndim != 2 or vertices_array.shape[1] != 2:
            raise ValueError("vertices must be a sequence of (x, y) pairs.")


def points_in_polygon(
    X: np.ndarray,
    Y: np.ndarray,
    vertices: Sequence[tuple[float, float]],
) -> np.ndarray:
    """
    Return Boolean mask for points inside a polygon using ray casting.

    X, Y:
        Meshgrid arrays with same shape.

    vertices:
        Sequence of polygon vertices in x-y order.
    """

    vertices_array = np.asarray(vertices, dtype=float)

    if vertices_array.ndim != 2 or vertices_array.shape[1] != 2:
        raise ValueError("vertices must be a sequence of (x, y) pairs.")

    n_vertices = len(vertices_array)

    if n_vertices < 3:
        raise ValueError("At least 3 vertices are required.")

    inside = np.zeros_like(X, dtype=bool)

    x_points = X
    y_points = Y

    x0 = vertices_array[-1, 0]
    y0 = vertices_array[-1, 1]

    for k in range(n_vertices):
        x1 = vertices_array[k, 0]
        y1 = vertices_array[k, 1]

        intersects = (
            ((y0 > y_points) != (y1 > y_points))
            & (
                x_points
                < (x1 - x0) * (y_points - y0) / (y1 - y0 + 1e-15) + x0
            )
        )

        inside ^= intersects

        x0 = x1
        y0 = y1

    return inside


def polygon_pillar_height_map(
    x: np.ndarray,
    y: np.ndarray,
    footprint: PolygonPillarFootprint,
    channel: ChannelGeometry2D | None = None,
) -> np.ndarray:
    """
    Build normalized pillar height map H_p(x, y) / H_c
    for a generic polygonal pillar footprint.

    Returns array with shape (len(y), len(x)).
    """

    footprint.validate()

    if channel is not None:
        channel.validate()

    X, Y = np.meshgrid(x, y, indexing="xy")

    if channel is not None:
        inside_channel = (
            (X >= channel.x_min)
            & (X <= channel.x_max)
            & (Y >= channel.y_min)
            & (Y <= channel.y_max)
        )
    else:
        inside_channel = np.ones_like(X, dtype=bool)

    inside_pillar = points_in_polygon(
        X=X,
        Y=Y,
        vertices=footprint.vertices,
    )

    height_map = np.zeros_like(X, dtype=float)
    height_map[inside_channel & inside_pillar] = footprint.height_ratio

    return height_map


def x_dependent_lateral_deflection(
    velocity_x: np.ndarray,
    velocity_y: np.ndarray,
    percentile: float = 95.0,
) -> np.ndarray:
    """
    Compute an x-dependent forcing signal from lubrication velocity.

    For each x-column, we estimate the characteristic lateral streamline
    slope |u_y/u_x| across y.
    """

    speed_x = np.maximum(np.abs(velocity_x), 1e-12)
    lateral_ratio = np.abs(velocity_y) / speed_x

    return np.percentile(lateral_ratio, percentile, axis=0)


def advect_cross_section_along_x(
    labels_yz: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    x: np.ndarray,
    forcing_x: np.ndarray,
    height_ratio: float,
    strength_y: float = 0.8,
    strength_z: float = 0.5,
    center_y: float = 0.5,
    width_ratio: float = 0.35,
    mount: str = "bottom",
    n_substeps: int = 1,
) -> np.ndarray:
    """
    March a y-z cross-sectional label field along x.

    The x-dependent forcing comes from the lubrication solver.
    At each x-location, we build an approximate y-z velocity field and
    use semi-Lagrangian backtracing to update the cross-section.

    labels_yz:
        Label field with shape (len(z), len(y)).
    """

    if labels_yz.shape != (len(z), len(y)):
        raise ValueError("labels_yz must have shape (len(z), len(y)).")

    if len(forcing_x) != len(x):
        raise ValueError("forcing_x must have same length as x.")

    if not 0.0 < height_ratio < 1.0:
        raise ValueError("height_ratio must be between 0 and 1.")

    Y, Z = np.meshgrid(y, z, indexing="xy")

    result = labels_yz.copy()

    dy_signed = Y - center_y

    lateral_envelope = np.exp(
        -(dy_signed**2) / (2.0 * width_ratio**2)
    )

    if mount == "bottom":
        inside_height = (Z >= 0.0) & (Z <= height_ratio)
        vertical_push = height_ratio - Z
    elif mount == "top":
        inside_height = (Z <= 1.0) & (Z >= 1.0 - height_ratio)
        vertical_push = Z - (1.0 - height_ratio)
    else:
        raise ValueError("mount must be 'bottom' or 'top'.")

    inside_height = inside_height.astype(float)

    dz_dx = np.gradient(x)

    for ix in range(len(x)):
        forcing = forcing_x[ix]

        for _ in range(n_substeps):
            dx_step = dz_dx[ix] / n_substeps

            influence = forcing * lateral_envelope * inside_height

            velocity_yz_y = (
                strength_y
                * dy_signed
                * influence
            )

            velocity_yz_z = (
                strength_z
                * vertical_push
                * influence
            )

            source_y = Y - velocity_yz_y * dx_step
            source_z = Z - velocity_yz_z * dx_step

            source_y = np.clip(source_y, y[0], y[-1])
            source_z = np.clip(source_z, z[0], z[-1])

            result = nearest_label_sample(
                labels=result,
                x=y,
                y=z,
                x_query=source_y,
                y_query=source_z,
            )

    return result
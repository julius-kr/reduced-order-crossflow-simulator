from __future__ import annotations

import numpy as np


def parabolic_velocity_1d(x: np.ndarray) -> np.ndarray:   #input: array of coordiantes between 0 and 1, output: array of velocity values at those coordinates
    """
    Normalized 1D Poiseuille-like velocity profile on [0, 1].

    u = 0 at walls, maximum at center.
    """
    return 6.0 * x * (1.0 - x)  # parabolic profile with u(0)=u(1)=0 and u(0.5)=1

def rectangular_channel_velocity_2d(    #calculates 2D velocity field for a rectangular channel 
    y: np.ndarray,                      #input: y is an array of coordinates between 0 and 1 along the width of the channel
    z: np.ndarray,                      #input: z is an array of coordinates between 0 and 1 along the height of the channel
    width: float = 1.0,                 #input: width is the physical width of the channel, default is 1.0
    height: float = 1.0,                 #input: height is the physical height of the channel, default is 1.0
    n_terms: int = 100,                   #input: n_terms is the number of terms to include in the Fourier series solution, default is 50; more terms will give a more accurate velocity field but will take more computation time
) -> np.ndarray:                         #output: a 2D array of velocity values at the coordinates defined by y and z
    """
    Compute a normalized fully developed laminar velocity field
    in a rectangular channel using a sine-series solution.

    The returned array has shape (len(z), len(y)) and represents u(y, z).

    The field is normalized so that the area-averaged velocity is 1.
    """

    if width <= 0 or height <= 0:        #safety check
        raise ValueError("width and height must be positive.")

    if n_terms <= 0:                     #safety check
        raise ValueError("n_terms must be positive.")

    y_physical = y * width             # physical coordinates = normalized coordinates * physical dimensions
    z_physical = z * height

    Y, Z = np.meshgrid(y_physical, z_physical, indexing="xy") # create 2D arrays Y and Z that contain the physical coordinates of each point in the grid defined by y and z
    velocity = np.zeros_like(Y, dtype=float)   # initialize a 2D array of the same shape as Y (and Z) to store the velocity values, initialized to zero, this is where we accumulate the Fourier series

    odd_modes = range(1, 2 * n_terms, 2)     
    """
     we only sum over odd modes (m and n are odd integers) because the sine series solution for a rectangular channel involves only odd harmonics to satisfy the no-slip boundary conditions at the walls. 
     The range(1, 2 * n_terms, 2) generates the sequence of odd integers starting from 1 up to (but not including) 2 * n_terms, which gives us the first n_terms odd integers: 1, 3, 5, ..., (2*n_terms - 1).
    """

    for m in odd_modes:
        sin_y = np.sin(m * np.pi * Y / width)  # compute the sine term for the y-direction for mode m; eliminates walls

        for n in odd_modes:
            sin_z = np.sin(n * np.pi * Z / height) # compute the sine term for the z-direction for mode n; eliminates walls

            coefficient = 16.0 / (                 
                np.pi**4
                * m
                * n
                * ((m / width) ** 2 + (n / height) ** 2)
            )
            """
            the coefficient for the Fourier series term corresponding to modes m and n, derived from the analytical solution of the Navier-Stokes equations for laminar flow in a rectangular channel; 
            the factor of 16 comes from the normalization and the sine series coefficients, and the denominator includes the mode numbers and the physical dimensions of the channel
            """

            velocity += coefficient * sin_y * sin_z

    total_flow = np.trapezoid(
        np.trapezoid(velocity, y, axis=1),   #double integration of u(y,z) first over y, then over z, from 0 to 1 in each case. axis=1 since y varies over columns. 
        z,
    )

    area = 1.0  #since normalized coordinates, the area of the cross-section is 1.0

    average_velocity = total_flow / area  #physically makes sense, since area=1, no numerical change

    if average_velocity == 0:   #safety check 
        raise ValueError("average velocity cannot be zero.")  

    return velocity / average_velocity # normalize the velocity field so that the area-averaged velocity is 1 


def effective_velocity_profile(           #reduces a 2D velocity field to a 1D effective velocity profile by integrating across one axis
    velocity_2d: np.ndarray,              #input: velocity_2d is a 2D array of velocity values
    coordinates: np.ndarray,               #input: coordinates is a 1D array of coordinates corresponding to one axis of the velocity field
    integrate_axis: int,                   #input: integrate_axis specifies which axis to integrate over
) -> np.ndarray:                          #output: a 1D array of effective velocity values that vary along the other axis
    """
    Reduce a 2D velocity field to a 1D effective velocity profile.

    integrate_axis = 1:
        integrate across y, result varies along z

    integrate_axis = 0:
        integrate across z, result varies along y
    """

    if velocity_2d.ndim != 2:         #safety check to ensure that velocity_2d is a 2D array                               
        raise ValueError("velocity_2d must be a 2D array.")

    return np.trapezoid(       # compute the integral of the 2D velocity field across the specified axis using the trapezoidal rule, 
        velocity_2d,
        coordinates,
        axis=integrate_axis,
    )


def cumulative_flow_fraction(x: np.ndarray, velocity: np.ndarray) -> np.ndarray:  #input: x is an array of coordinates, velocity is an array of velocity values at those coordinates, output: cumulative flow fraction at each coordinate
    """
    Compute cumulative flow fraction from a velocity profile.

    Returns values from 0 to 1.
    """
    if x.ndim != 1 or velocity.ndim != 1:  # safety checks to ensure that x and velocity are 1D arrays
        raise ValueError("x and velocity must be 1D arrays.")

    if len(x) != len(velocity):            #safety check to ensure that x and velocity have the same length, which is necessary for element-wise operations
        raise ValueError("x and velocity must have the same length.")

    if np.any(velocity < 0):               # safety check to ensure that velocity is non-negative
        raise ValueError("velocity must be non-negative.")

    dx = np.diff(x)                       # compute the differences between consecutive x values, which represent the widths of the intervals between the coordinates

    cumulative = np.zeros_like(x, dtype=float)   # initialize an array of the same shape as x to store the cumulative flow fraction values, initialized to zero
    cumulative[1:] = np.cumsum(
        0.5 * (velocity[:-1] + velocity[1:]) * dx
    )

    total_flow = cumulative[-1]

    if total_flow == 0:
        raise ValueError("total flow cannot be zero.")

    return cumulative / total_flow


def boundaries_from_flow_rates(
    x: np.ndarray,                      #input: x is an array of coordinates between 0 and 1
    flow_rates: list[float],            #input: flow_rates is a list of flow rates for each layer
    velocity: np.ndarray | None = None, #input: velocity is an optional array of velocity values at the coordinates in x; if not provided, it will be computed using the parabolic_velocity_1d function
) -> list[float]:                       #output: a list of layer boundaries, including the start (0.0) and end (1.0) of the domain, and the intermediate boundaries that separate the layers 
    """
    Compute layer boundaries so each layer carries the requested flow fraction.
    """
    rates = np.asarray(flow_rates, dtype=float)

    if len(rates) == 0:                 #safety check 
        raise ValueError("flow_rates must not be empty.")

    if np.any(rates <= 0):              #safety check
        raise ValueError("All flow rates must be positive.")

    if velocity is None:                #if velocity is not provided, compute it using the parabolic_velocity_1d function
        velocity = parabolic_velocity_1d(x)

    fractions = rates / rates.sum()     # compute the flow fraction for each layer
    targets = np.cumsum(fractions)[:-1]   # compute the cumulative flow fraction targets for the layer boundaries, excluding the last one since it corresponds to the end of the domain

    cumulative = cumulative_flow_fraction(x, velocity)  # compute the cumulative flow fraction at each coordinate in x using the cumulative_flow_fraction function

    boundaries = [0.0]   # initialize the list of boundaries with the start of the domain (0.0)

    for target in targets:   #loop over each target cumulative flow fraction for the layer boundaries
        boundary = np.interp(target, cumulative, x) # use np.interp to find the coordinate in x that corresponds to the target cumulative flow fraction, by interpolating between the values in cumulative and x
        boundaries.append(float(boundary))  # append the computed boundary to the list of boundaries

    boundaries.append(1.0)

    return boundaries
#this module defines the CrossSection class, which represents a 2D channel cross-section. The grid is stored as a 2D NumPy array of labels, where each value corresponds to a stream/material ID. The class provides methods for initializing the grid, filling it with a specific label, copying the cross-section, setting rectangular regions with specific labels, and retrieving unique labels present in the grid.
from dataclasses import dataclass
import numpy as np


@dataclass
class CrossSection:
    """
    Represents a 2D channel cross-section.

    The grid is stored as labels[z, y], where:
    - z is the vertical direction
    - y is the horizontal direction
    - each value is a stream/material ID
    """

    ny: int = 300  # number of grid points in the y direction i.e. horizontal
    nz: int = 300  # number of grid points in the z direction i.e. vertical
    default_label: int = 0  # default stream/material ID for the entire cross-section

    """ 
    cs = CrossSection() creates 300 X 300 grid with all labels set to 0
    cs= CrossSection(ny=100, nz=80, default_label=3) creates 80 X 100 grid with all labels set to 3
    """
    

    def __post_init__(self) -> None:  # called after the dataclass is initialized
        if self.ny <= 0 or self.nz <= 0:
            raise ValueError("ny and nz must be positive integers.") #safety check to ensure valid grid dimensions

        self.labels = np.full(    #create a 2D array of shape (nz, ny) filled with default_label
            shape=(self.nz, self.ny),
            fill_value=self.default_label,
            dtype=int,
        )

        """
        np.full(shape=(3, 4), fill_value=5) creates a 3x4 array filled with the value 5:
        [[5 5 5 5]
         [5 5 5 5]
         [5 5 5 5]]
         In our case, shape=(self.nz, self.ny) creates an array with nz rows and ny columns, and fill_value=self.default_label fills it with the specified default label.
         dtype=int ensures that the array elements are of integer type, which is appropriate for stream/material IDs.
        """

        self.y = np.linspace(0.0, 1.0, self.ny)  # create a 1D array of length ny with values linearly spaced between 0 and 1, representing the horizontal coordinates of the grid points
        self.z = np.linspace(0.0, 1.0, self.nz)  # create a 1D array of length nz with values linearly spaced between 0 and 1, representing the vertical coordinates of the grid points

        """
        y= 0 left wall, y=1 right wall, z=0 bottom wall, z=1 top wall"""

    @property
    def shape(self) -> tuple[int, int]:  # returns the shape of the labels array, which is (nz, ny)
        return self.labels.shape

    def fill(self, label: int) -> None:  # fill the entire labels array with the specified label
        self.labels[:, :] = label

    def copy(self) -> "CrossSection":  # create a new CrossSection instance with the same dimensions and default label, and copy the labels array to it
        new_section = CrossSection(
            ny=self.ny,
            nz=self.nz,
            default_label=self.default_label,
        )
        new_section.labels = self.labels.copy() # create a copy of the labels array to ensure that the new instance is independent of the original
        return new_section

    def set_region(
        self,
        y_min: float,
        y_max: float,
        z_min: float,
        z_max: float,
        label: int,
    ) -> None:
        """
        Set a rectangular region in normalized coordinates [0, 1] x [0, 1].
        """

        if not (0.0 <= y_min <= y_max <= 1.0):
            raise ValueError("Expected 0 <= y_min <= y_max <= 1.")
        if not (0.0 <= z_min <= z_max <= 1.0):
            raise ValueError("Expected 0 <= z_min <= z_max <= 1.")

        y_mask = (self.y >= y_min) & (self.y <= y_max)
        z_mask = (self.z >= z_min) & (self.z <= z_max)

        self.labels[np.ix_(z_mask, y_mask)] = label

    def unique_labels(self) -> np.ndarray:
        return np.unique(self.labels)
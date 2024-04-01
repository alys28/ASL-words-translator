import numpy as np
import torch
import math
from typing import Optional


class ProjectiveGeometry:
    """Perform projective geometry on a set of 2D or 3D points
    """
    values: torch.Tensor
    points: np.array  # each tuple is (x, y, z)
    pt_fuite: tuple[float, float, float]  # (x, y, z)
    x: np.array  # only the x values.
    z: np.array

    def __init__(self, values: torch.Tensor) -> None:
        """Initialize the points in the tensors"""
        self.values = values


        # still need to define these values from the torch object
        self.x = np.array([point[0] for points in self.points])
        self.z = np.array([point[0] for points in self.points])

        compute_pt_fuite_x()

    def compute_pt_fuite_x(self, negativity: bool = True) -> None:
        """Initialize the point the fuite, with x being the mean, and y being random"""
        # initialize as mean, but could also be initialized randomly.
        self.pt_fuite[0] = np.mean(self.x)
        self.pt_fuite[2] = np.mean(self.z)

        # these values can be updated later.
        if (negativity == True):
            self.pt_fuite[1] = np.random.uniform(-30, -5)
        else:
            self.pt_fuite[1] = np.random.uniform(5, 15)

    def projective_geometry(self) -> np.array:
        """Given a pt_fuite tuple[x, y], returns an array of points"""
        ax = pt_fuite[0]
        by = pt_fuite[1]

        new_points = []
        for i in range(len(self.points)):
            slope = -by / (self.points[0][i] - ax)
            # unsure about this function.
            x_coord = (self.points[1][i] - by *
                       (1 + (ax / (self.points[0][i] - ax)))) / slope
            new_x.append((x_coord, self.points[1], self.points[2]))
        return np.array(new_points)

    def visualize(values: torch.Tensor) -> None:
        """Visualize the new coordinates
        """
        pass


def compute_ProjectiveGeometry(values: torch.Tensor) -> torch.Tensor:
    """Compute the projective geometry given a set of points"""
    # I need the x, y, and z channels.
    # 120 frames.
    # technically, I personally need to show it each file, the x, y and z
    values = torch.reshape(values, (120, 3, 25, 2))

    final = []
    # call the ProjectiveGeometry object for each frame
    for frame in values:
    # calling the ProjectiveGeometry Object.
        pg = ProjectiveGeometry(values)
        final.append(pg.projective_geometry())
    final = torch.tensor(final)
    return torch.reshape(final, (3, 120, 25, 2))


if __name__ == '__main__':
    import torch

    # Load the file
    pt_file = torch.load("sequence.pt")

    # Print the head of the file
    print(type(pt_file))

    compute_ProjectiveGeometry(pt_file)

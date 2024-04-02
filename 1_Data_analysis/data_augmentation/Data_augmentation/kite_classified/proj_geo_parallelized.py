# apply the same vanishing points for all frames in a video
import numpy as np
import torch
import math
from typing import Optional
import random


class ProjectiveGeometry:
    """Perform projective geometry on a set of 2D or 3D points
    """
    points: torch.Tensor  # each tuple is (x, y, z)
    pt_fuite: list  # (x, y, z)

    def __init__(self, points: torch.Tensor) -> None:
        """Initialize the points in the tensors"""
        # (3, 2, 120, 25)
        self.points = points
        self.pt_fuite = [0, 0, 0]
        self.compute_pt_fuite_x(random.choice([True, False]))

    def compute_pt_fuite_x(self, negativity: bool = True) -> None:
        """Initialize the point the fuite, with x being the mean, and y being random"""
        # initialize as mean, but could also be initialized randomly.
        self.pt_fuite[0] = torch.mean(self.points[0])
        self.pt_fuite[2] = torch.mean(self.points[2])

        # these values can be updated later.
        if negativity:
            self.pt_fuite[1] = np.random.uniform(-30, -5)
        else:
            self.pt_fuite[1] = np.random.uniform(-30, -5)

    def projective_geometry(self) -> torch.tensor:
        """Given a pt_fuite tuple[x, y], returns an array of points"""
        ax = self.pt_fuite[0]
        by = self.pt_fuite[1]
        cz = self.pt_fuite[2]

        new_points = self.points.clone()
        # compute for each joint
        # print('debug', self.points)
        # print(self.points.size()[1])

        # parallelized
        slope = -by / (self.points[0] - ax)
        # this would take the 0th index of all the columns
        new_points[0] = (self.points[1] - by *
                         (1 + (ax / (self.points[0] - ax)))) / slope

        # changing the z_axis
        slope = -by / (self.points[2] - cz)
        new_points[2] = (self.points[1] - by *
                         (1 + (cz / (self.points[2] - cz)))) / slope
        return new_points

    def visualize(values: torch.Tensor) -> None:
        """Visualize the new coordinates
        """
        pass


def compute_ProjectiveGeometry(inp_tensor: torch.Tensor) -> torch.Tensor:
    """Compute the projective geometry given a set of points"""
    inp_tensor = torch.reshape(inp_tensor, (3, 2, 120, 25))
    pg = ProjectiveGeometry(inp_tensor)
    output = pg.projective_geometry()
    return torch.reshape(output, (3, 120, 25, 2))


if __name__ == '__main__':
    import torch

    # Load the file
    pt_file = torch.load("sequence.pt")

    # Print x coordinates
    print(pt_file.data[0][0][0])
    print(compute_ProjectiveGeometry(pt_file.data)[0][0][0])

    # printing y coordinates.
    print(pt_file.data[1][0][0])
    print(compute_ProjectiveGeometry(pt_file.data)[1][0][0])

    # printing z coordinate:
    print(pt_file.data[2][0][0])
    print(compute_ProjectiveGeometry(pt_file.data)[2][0][0])
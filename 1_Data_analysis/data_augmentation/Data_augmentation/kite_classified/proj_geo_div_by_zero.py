import numpy as np
import torch
import math
from typing import Optional
import random

class ProjectiveGeometry:
    """Perform projective geometry on a set of 3D points
    """
    points: torch.Tensor  # each tuple is (x, y, z)
    pt_fuite: list  # (x, y, z)

    def __init__(self, points: torch.Tensor) -> None:
        """Initialize the points in the tensors"""
        # (3, 2, 120, 25)
        self.points = points
        self.pt_fuite = [0, 0, 0]
        self.compute_pt_fuite_x()

    def compute_pt_fuite_x(self) -> None:
        """Initialize the point the fuite, with x being the mean, and y being random"""
        # initialize as mean, but could also be initialized randomly.
        self.pt_fuite[0] = torch.mean(self.points[0])
        self.pt_fuite[2] = torch.mean(self.points[2])

        lower_y = torch.max(self.points[1]) 
        upper_y = torch.max(self.points[1]) + torch.abs(torch.min(self.points[1]))

        self.pt_fuite[1] = lower_y  + 1.0 + (upper_y - lower_y) * torch.rand(1)
        # self.pt_fuite[1] = lower_y
        
        if self.pt_fuite[1] == 0:
            compute_pt_fuite_x()


    def projective_geometry(self) -> torch.tensor:
        """Given a pt_fuite tuple[x, y], returns an array of points"""
        ax = self.pt_fuite[0]
        by = self.pt_fuite[1]
        cz = self.pt_fuite[2]
        
        new_points = self.points.clone()
        
        slope = - (self.points[0] - ax) / by
        new_points[0] = (self.points[1] - by *
                         (1 + (ax / (self.points[0] - ax)))) * slope

        slope = - (self.points[2] - cz) / by
        new_points[2] = (self.points[1] - by *
                         (1 + (cz / (self.points[2] - cz)))) * slope
        return new_points

def compute_projective_geometry(inp_tensor: torch.Tensor, p=1) -> torch.Tensor:
    """Compute the projective geometry given a set of points"""
    if random.random() < p:
        pg = ProjectiveGeometry(inp_tensor)
        output = pg.projective_geometry()
        return output
    else:
        return inp_tensor

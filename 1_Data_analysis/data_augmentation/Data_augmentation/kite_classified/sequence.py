# dimensions are 3, 120, 25, 2
import torch

# Load the file
pt_file = torch.load("sequence.pt")

# Print the head of the file
print(pt_file[:])
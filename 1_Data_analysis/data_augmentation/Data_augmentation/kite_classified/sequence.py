# dimensions are 3, 120, 25, 2
import torch

# Load the file
pt_file = torch.load("sequence.pt")

# Print the head of the file
print(pt_file[:])

# torch.Size([3, 120, 25, 2])
# channels, x, y, z
# 120 frames
# 25 joints
# 2 values.
print(pt_file.size())
pt_file = torch.reshape(pt_file, (120, 3, 25, 2))
print(pt_file)
print(pt_file.size())


x = torch.tensor([[1,2,3], [2, 3, 4]])
print(x.size())

x = torch.reshape(x, (3, 2))
print(x)
print(x.size())

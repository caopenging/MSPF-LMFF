# import torch
# from torch import nn
#
#
# class C_lenet(nn.Module):
#     def __init__(self):
#         super(C_lenet, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(1, 16, 8),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(16, 32, 8),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(1, 16, 8),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(16, 32, 8),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2)
#         )
#         self.fc_x2 = nn.Linear(1,128)
#         self.fc1_x2 = nn.Linear(128,1024)
#         self.fc2_x2 = nn.Linear(1024,2048)
#         self.fc1 = nn.Linear(1, 1)
#         self.fc2 = nn.Linear(1, 1)
#         self.fc3 = nn.Linear(1, 256)
#
#     def forward(self, x1, x2):
#         x1 = self.conv1(x1.reshape(1,x1.shape[0],x1.shape[1],x1.shape[2]))
#         x2 = self.fc_x2(x2)
#         x2 = self.fc1_x2(x2)
#         x2 = self.fc2_x2(x2)
#         x2 = self.conv2(x2.reshape(1,x2.shape[0],x2.shape[1],x2.shape[2]))
#         x1 = torch.flatten(x1, 1)
#         x2 = torch.flatten(x2, 1)
#         x = torch.cat((x1, x2),1)
#         x = self.fc1(x)  # output(120)
#         x = self.fc2(x) # output(84)
#         x = self.fc3(x)
#         return x
import torch
from torch import nn

class C_lenet(nn.Module):
    def __init__(self):
        super(C_lenet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_x2 = nn.Linear(420992, 128)  # Assuming 420992 is the size of x2 after flattening
        self.fc1_x2 = nn.Linear(128, 1024)
        self.fc2_x2 = nn.Linear(1024, 2048)
        self.fc1 = nn.Linear(782912 + 2048, 120)  # Adjust input size to match concatenated output of x1 and x2
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 256)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x1 = torch.flatten(x1, 1)

        x2 = self.fc_x2(x2)
        x2 = self.fc1_x2(x2)
        x2 = self.fc2_x2(x2)

        x = torch.cat((x1, x2), 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

# Assuming x1 has shape (1, 1, 688, 611) and x2 has shape (1, 420992)
# Example usage:
model = C_lenet()
x1 = torch.randn(1, 1, 688, 611)  # Example input for x1
x2 = torch.randn(1, 420992)       # Example input for x2
output = model(x1, x2)
print(output.shape)  # Check output shape

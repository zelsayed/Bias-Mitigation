import torch
import torch.nn as nn
import torch.nn.functional as F

class ModifiedLeNet(nn.Module):
    def __init__(self, in_channels, out_dims):
        """
        Initializes the ModifiedLeNet model.

        Args:
            in_channels (int): Number of input channels for the convolutional layers.
            out_dims (int): Number of output dimensions (e.g., number of classes).
        """
        super(ModifiedLeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.color_dim = 3
        self.fc_color = nn.Linear(84 + self.color_dim, out_dims)

    def forward(self, x, color_vector):
        """
        Defines the forward pass of the ModifiedLeNet model.

        Args:
            x (torch.Tensor): Input image tensor.
            color_vector (torch.Tensor): Input color vector tensor.

        Returns:
            torch.Tensor: Output logits from the model.
        """
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        combined_vector = torch.cat((x, color_vector), dim=1)
        output = self.fc_color(combined_vector)
        return output

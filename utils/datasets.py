import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

class ColoredDataset(Dataset):
    def __init__(self, dataset, classes=None, colors=[0, 1], std=0):
        """
        Initializes the ColoredDataset.

        Args:
            dataset (torch.utils.data.Dataset): The original dataset (e.g., MNIST).
            classes (int, optional): Number of classes in the dataset.
            colors (list or torch.Tensor, optional): Color ranges or specific colors for each class.
            std (float, optional): Standard deviation for color perturbation.
        """
        self.dataset = dataset
        self.colors = colors
        self.gray_transform = transforms.Grayscale(num_output_channels=1)
        if classes is None:
            classes = max([y for _, y in dataset]) + 1
        if isinstance(colors, torch.Tensor):
            self.colors = colors
        elif isinstance(colors, list):
            self.colors = torch.Tensor(classes, 3, 1, 1).uniform_(colors[0], colors[1])
        else:
            raise ValueError('Unsupported colors!')
        self.perturb = std * torch.randn(len(self.dataset), 3, 1, 1)
        
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img_gray = self.gray_transform(img)
        color_vector = (self.colors[label] + self.perturb[idx]).clamp(0, 1).mean(dim=(1, 2))
        return img_gray, color_vector, label

def visualize_samples(loader, title):
    """
    Visualizes a batch of samples from the given DataLoader.

    Args:
        loader (DataLoader): DataLoader from which to get the samples.
        title (str): Title for the visualization plot.
    """
    batch = next(iter(loader))
    images, color_vectors, labels = batch
    fig, axs = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle(title)
    for i, ax in enumerate(axs.flatten()):
        if i >= len(images):
            break
        img_gray = images[i].squeeze().cpu().numpy()
        img_color = np.stack([img_gray] * 3, axis=-1)
        color = color_vectors[i].cpu().numpy()
        img_color *= color.reshape(1, 1, 3)
        label = labels[i].item()
        ax.imshow(img_color)
        ax.set_title(f'Label: {label}')
        ax.axis('off')
    plt.show()

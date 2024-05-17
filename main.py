import torch
import argparse
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from utils.datasets import ColoredDataset, visualize_samples
from utils.models import ModifiedLeNet
from utils.measure import measure_generalization
from regularization import RegParameters, Perturbation, Regularization

def main(args):
    """
    Main function to set up the dataset, model, and training process.

    Args:
        args (argparse.Namespace): Command line arguments.
    """
    torch.cuda.set_device(args.gpu)
    
    reg_params = RegParameters(
        lambda_image=args.lambda_image, 
        lambda_color=args.lambda_color, 
        n_samples=args.n_samples, 
        norm=args.norm, 
        optim_method=args.optim_method, 
        estimation=args.estimation,
        knn_value=args.knn_value
    )

    train_set = datasets.MNIST(args.data_dir, train=True, download=True, transform=transforms.ToTensor())
    test_set = datasets.MNIST(args.data_dir, train=False, download=True, transform=transforms.ToTensor())

    train_size = int(0.8 * len(train_set))
    val_size = len(train_set) - train_size
    train_subset, val_subset = random_split(train_set, [train_size, val_size])

    colored_train_set = ColoredDataset(train_subset, classes=10, colors=[0, 1], std=args.std)
    train_loader = DataLoader(colored_train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    colored_val_set = ColoredDataset(val_subset, classes=10, colors=colored_train_set.colors, std=args.std)
    val_loader_colored = DataLoader(colored_val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    grayscale_val_set = ColoredDataset(val_subset, classes=10, colors=[1, 1], std=0)
    val_loader_grayscale = DataLoader(grayscale_val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    colored_test_set = ColoredDataset(test_set, classes=10, colors=colored_train_set.colors, std=args.std)
    test_loader_colored = DataLoader(colored_test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    grayscale_test_set = ColoredDataset(test_set, classes=10, colors=[1, 1], std=0)
    test_loader_grayscale = DataLoader(grayscale_test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Optional: Visualize Samples
    # visualize_samples(train_loader, "Training Dataset Samples")
    # visualize_samples(val_loader_colored, "Colored Validation Dataset Samples")
    # visualize_samples(val_loader_grayscale, "Grayscale Validation Dataset Samples")
    # visualize_samples(test_loader_colored, "Colored Test Dataset Samples")
    # visualize_samples(test_loader_grayscale, "Grayscale Test Dataset Samples")

    model = ModifiedLeNet(1, 10)
    best_val_acc_grayscale, best_test_acc_colored, best_test_acc_grayscale, max_test_acc_grayscale_any = measure_generalization(
        train_loader, val_loader_colored, val_loader_grayscale, [test_loader_colored, test_loader_grayscale], 
        model, epochs=args.epochs, lr=args.lr, reg_params=reg_params, patience=args.patience
    )
    
    print(f"Test accuracy on Colored MNIST at convergence: {best_test_acc_colored:.2f}%")
    print(f"Max Accuracy Observed on Grayscale MNIST: {max_test_acc_grayscale_any:.2f}%")
    print(f"Accuracy Observed on Grayscale MNIST at Convergence: {best_test_acc_grayscale:.2f}%")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training a Modified LeNet on Colored MNIST with Regularization.')
    parser.add_argument('--gpu', default=0, type=int, help='GPU ID to use.')
    parser.add_argument('--data_dir', default='./data', type=str, help='Directory to download the MNIST dataset.')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size for training and evaluation.')
    parser.add_argument('--epochs', default=50, type=int, help='Number of training epochs.')
    parser.add_argument('--lr', default=1e-2, type=float, help='Learning rate for the optimizer.')
    parser.add_argument('--lambda_image', default=1e-12, type=float, help='Regularization parameter for image perturbations.')
    parser.add_argument('--lambda_color', default=7e-11, type=float, help='Regularization parameter for color vector perturbations.')
    parser.add_argument('--n_samples', default=10, type=int, help='Number of samples for perturbation.')
    parser.add_argument('--norm', default=2.0, type=float, help='Norm for regularization.')
    parser.add_argument('--optim_method', default='max_ent', type=str, help='Optimization method for regularization.')
    parser.add_argument('--estimation', default='var', type=str, help='Estimation method for regularization.')
    parser.add_argument('--knn_value', default=219, type=int, help='K-NN value for perturbation tensor.')
    parser.add_argument('--std', default=0.1, type=float, help='Standard deviation for color perturbation.')
    parser.add_argument('--patience', default=10, type=int, help='Patience for early stopping.')
    
    args = parser.parse_args()
    main(args)

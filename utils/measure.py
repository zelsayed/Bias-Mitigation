import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from regularization import Perturbation, Regularization, RegParameters

def measure_generalization(train_loader, val_loader_colored, val_loader_grayscale, test_loaders, model, epochs, lr, reg_params=None, patience=10, model_save_path="best_model.pth"):
    """
    Trains and evaluates the model, measuring generalization performance.

    Args:
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        val_loader_colored (torch.utils.data.DataLoader): DataLoader for colored validation data.
        val_loader_grayscale (torch.utils.data.DataLoader): DataLoader for grayscale validation data.
        test_loaders (list of torch.utils.data.DataLoader): List of DataLoaders for testing data.
        model (nn.Module): The model to be trained.
        epochs (int): Number of epochs to train.
        lr (float): Learning rate for the optimizer.
        reg_params (RegParameters, optional): Regularization parameters.
        patience (int, optional): Number of epochs to wait for early stopping.
        model_save_path (str, optional): Path to save the best model.

    Returns:
        tuple: Best validation accuracy (grayscale), best test accuracy (colored), best test accuracy (grayscale), max test accuracy (grayscale) at any point.
    """
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    best_val_acc_grayscale = 0
    best_test_acc_colored = 0
    best_test_acc_grayscale = 0
    max_test_acc_grayscale_any = 0
    best_model = None
    best_epoch = 0

    train_losses = []
    train_accuracies = []
    val_accuracies_colored = []
    val_accuracies_grayscale = []
    test_accuracies_colored = []
    test_accuracies_grayscale = []
    
    fisher_image_props = []
    fisher_color_props = []
    
    no_improvement_epochs = 0

    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w

    model.train()
    with tqdm(total=epochs, desc="Training Model") as pbar:
        for epoch in range(epochs):
            correct = 0
            total = 0
            epoch_loss = 0
            fisher_image = 0
            fisher_color = 0
            for data in train_loader:
                images, color_vectors, labels = data
                images, color_vectors, labels = images.cuda(), color_vectors.cuda(), labels.cuda()
                
                optimizer.zero_grad()
                outputs = model(images, color_vectors)
                loss = F.cross_entropy(outputs, labels)
                
                if reg_params and (reg_params.lambda_image or reg_params.lambda_color):
                    expanded_logits = Perturbation.get_expanded_logits(outputs, reg_params.n_samples)
                    inf_image = Perturbation.perturb_tensor(images, reg_params.n_samples, reg_params.knn_value)
                    inf_color_vectors = Perturbation.perturb_tensor(color_vectors, reg_params.n_samples, reg_params.knn_value)

                    inf_output = model(inf_image, inf_color_vectors)
                    inf_loss = F.binary_cross_entropy_with_logits(expanded_logits, inf_output)

                    gradients = torch.autograd.grad(inf_loss, [inf_image, inf_color_vectors], create_graph=True)
                    grads = [Regularization.get_batch_norm(gradient, loss=inf_loss, estimation=reg_params.estimation) for gradient in gradients]
                    
                    reg_image = 0
                    reg_color_vectors = 0

                    if reg_params.lambda_image:
                        fisher_image = Regularization.get_regularization_term(grads[0], norm=reg_params.norm, optim_method=reg_params.optim_method)
                        reg_image = fisher_image * reg_params.lambda_image
                    if reg_params.lambda_color:
                        fisher_color = Regularization.get_regularization_term(grads[1], norm=reg_params.norm, optim_method=reg_params.optim_method)
                        reg_color_vectors = fisher_color * reg_params.lambda_color
                    
                    loss += reg_image + reg_color_vectors
                
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            # Compute Fisher information proportions
            total_fisher = fisher_image + fisher_color
            if total_fisher > 0:
                fisher_image_prop = fisher_image / total_fisher
                fisher_color_prop = fisher_color / total_fisher
            else:
                fisher_image_prop = 0
                fisher_color_prop = 0

            fisher_image_props.append(fisher_image_prop)
            fisher_color_props.append(fisher_color_prop)
            
            epoch_loss /= len(train_loader)
            train_losses.append(epoch_loss)
            train_accuracy = 100 * correct / total
            train_accuracies.append(train_accuracy)

            val_acc_colored = validate_model(val_loader_colored, model)
            val_acc_grayscale = validate_model(val_loader_grayscale, model)
            val_accuracies_colored.append(val_acc_colored)
            val_accuracies_grayscale.append(val_acc_grayscale)
            if val_acc_grayscale > best_val_acc_grayscale:
                best_val_acc_grayscale = val_acc_grayscale
                best_model = model.state_dict()
                best_epoch = epoch
                no_improvement_epochs = 0

                torch.save(best_model, model_save_path)

            else:
                no_improvement_epochs += 1

            model.load_state_dict(best_model)
            model.eval()
            test_acc_colored, test_acc_grayscale = [], []
            for loader in test_loaders:
                total = 0
                correct = 0
                with torch.no_grad():
                    for images, color_vectors, labels in loader:
                        images, color_vectors, labels = images.cuda(), color_vectors.cuda(), labels.cuda()
                        outputs = model(images, color_vectors)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                accuracy = 100 * correct / total
                if loader == test_loaders[0]:
                    test_acc_colored.append(accuracy)
                else:
                    test_acc_grayscale.append(accuracy)
            best_test_acc_colored = max(best_test_acc_colored, max(test_acc_colored))
            best_test_acc_grayscale = max(best_test_acc_grayscale, max(test_acc_grayscale))
            max_test_acc_grayscale_any = max(max_test_acc_grayscale_any, best_test_acc_grayscale)
            
            pbar.set_description(f"Epoch {epoch+1}/{epochs}")
            pbar.update()
            
            if epoch >= patience:
                smoothed_losses = moving_average(train_losses, patience)
                if len(smoothed_losses) >= 2 and abs(smoothed_losses[-1] - smoothed_losses[-2]) < 1e-3:
                    break
    
    model.load_state_dict(best_model)
    model.eval()
    for loader in test_loaders:
        total = 0
        correct = 0
        with torch.no_grad():
            for images, color_vectors, labels in loader:
                images, color_vectors, labels = images.cuda(), color_vectors.cuda(), labels.cuda()
                outputs = model(images, color_vectors)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        if loader == test_loaders[0]:
            test_accuracies_colored.append(accuracy)
        else:
            test_accuracies_grayscale.append(accuracy)
    
    max_test_acc_grayscale_any = max(max_test_acc_grayscale_any, max(test_accuracies_grayscale))

    # Convert tensors to CPU before plotting
    fisher_image_props = [prop.cpu().item() for prop in fisher_image_props]
    fisher_color_props = [prop.cpu().item() for prop in fisher_color_props]

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Train Loss Over Epochs')
    plt.show()

    plt.figure(figsize=(10, 5))
    epochs_range = range(1, len(train_accuracies) + 1)
    plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
    plt.plot(epochs_range, val_accuracies_colored, label='Validation Accuracy (Colored)')
    plt.plot(epochs_range, val_accuracies_grayscale, label='Validation Accuracy (Grayscale)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Train, Validation (Colored and Grayscale) Accuracy Over Epochs')
    plt.show()

    # Plot Fisher information proportions
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, fisher_image_props, label='Fisher Information Proportion (Image)')
    plt.plot(epochs_range, fisher_color_props, label='Fisher Information Proportion (Color)')
    plt.xlabel('Epoch')
    plt.ylabel('Fisher Information Proportion')
    plt.legend()
    plt.title('Fisher Information Proportions Over Epochs')
    plt.show()

    # Bar chart for Fisher information proportions in the first and last epoch
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_width = 0.35
    index = np.arange(2)
    
    fisher_first_epoch = [fisher_image_props[0], fisher_color_props[0]]
    fisher_last_epoch = [fisher_image_props[-1], fisher_color_props[-1]]
    
    bar1 = ax.bar(index, fisher_first_epoch, bar_width, label='First Epoch')
    bar2 = ax.bar(index + bar_width, fisher_last_epoch, bar_width, label='Last Epoch')
    
    ax.set_xlabel('Modality')
    ax.set_ylabel('Fisher Information Proportion')
    ax.set_title('Fisher Information Proportion by Modality in First and Last Epoch')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(['Image', 'Color'])
    ax.legend()

    plt.show()
    
    return best_val_acc_grayscale, best_test_acc_colored, best_test_acc_grayscale, max_test_acc_grayscale_any

def validate_model(val_loader, model):
    """
    Validates the model on the validation dataset.

    Args:
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        model (nn.Module): The model to be validated.

    Returns:
        float: Validation accuracy in percentage.
    """
    total = 0
    correct = 0
    with torch.no_grad():
        for images, color_vectors, labels in val_loader:
            images, color_vectors, labels = images.cuda(), color_vectors.cuda(), labels.cuda()
            outputs = model(images, color_vectors)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

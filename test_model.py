import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Import your architecture and data loader
from model import BasicCNN
from data_loader import get_loaders

def imshow(img, title):
    # Un-normalize
    # image = (image * std) + mean
    img = img / 2 + 0.5     
    npimg = img.numpy()
    
    # Transpose dimensions from (C, H, W) to (H, W, C) for Matplotlib
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.axis('off')

def visualize_predictions(model_path='cifar10_cnn.pth', num_images=5):
    # 1. Setup Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Testing on: {device}")

    # 2. Load Model
    model = BasicCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. Get Test Data
    _, test_loader = get_loaders(batch_size=num_images, use_subset=False)
    
    # 4. Get one batch of images
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # Move to device for prediction
    images_device = images.to(device)
    outputs = model(images_device)
    _, predicted = torch.max(outputs, 1)

    # 5. Class Names (CIFAR-10)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')

    # 6. Plotting
    fig = plt.figure(figsize=(15, 3))
    print("\n--- Model Predictions ---")
    
    for i in range(num_images):
        ax = fig.add_subplot(1, num_images, i + 1)
        
        # Get raw image (CPU)
        img = images[i]
        
        # Get labels
        true_label = classes[labels[i]]
        pred_label = classes[predicted[i]]
        
        # Color code: Green if correct, Red if wrong
        color = 'green' if true_label == pred_label else 'red'
        title = f"True: {true_label}\nPred: {pred_label}"
        
        # Un-normalize logic for visualization (approximate)
        # We revert the specific CIFAR-10 stats: (img * std) + mean
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
        img = img * std + mean
        
        # Clamp to [0, 1] to avoid matplotlib warnings
        img = torch.clamp(img, 0, 1)
        
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        ax.set_title(title, color=color, fontsize=12, fontweight='bold')
        ax.axis('off')
        
    plt.show()

if __name__ == '__main__':
    visualize_predictions()
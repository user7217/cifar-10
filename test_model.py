import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# --- CHANGE 1: Import the Correct Model Class ---
from model import ImprovedCNN 
from data_loader import get_loaders

def visualize_predictions(model_path='improved_cnn.pth', num_images=5): # --- CHANGE 2: Default path
    # 1. Setup Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Testing on: {device}")

    # 2. Load Model
    # --- CHANGE 3: Initialize the Correct Architecture ---
    model = ImprovedCNN().to(device)
    
    # Load weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded weights from {model_path}")
    except FileNotFoundError:
        print(f"Error: Could not find {model_path}. Make sure you ran train.py first.")
        return
    except RuntimeError as e:
        print(f"Error: Architecture mismatch. Are you using ImprovedCNN with basic_cnn.pth weights? \nDetails: {e}")
        return

    model.eval()

    # 3. Get Test Data
    # Note: get_loaders returns (train, test). We only need test.
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
        
        img = images[i]
        true_label = classes[labels[i]]
        pred_label = classes[predicted[i]]
        
        color = 'green' if true_label == pred_label else 'red'
        title = f"True: {true_label}\nPred: {pred_label}"
        
        # Un-normalize logic for visualization
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
        img = img * std + mean
        img = torch.clamp(img, 0, 1)
        
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        ax.set_title(title, color=color, fontsize=12, fontweight='bold')
        ax.axis('off')
        
    plt.show()

if __name__ == '__main__':
    visualize_predictions()
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

def get_loaders(batch_size=128, use_subset=False):
    
    # Standard CIFAR-10 mean and std deviation
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    
    # Define transforms: Convert to Tensor and Normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    # Download and load datasets
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
    
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)

    # Subset logic for assignment constraints
    if use_subset:
        indices = range(10000)
        train_set = Subset(train_set, indices)
        print("Using Subset: 10,000 images")
    else:
        print("Using Full Dataset: 50,000 images")

    # Create DataLoaders
    # num_workers=2 is optimized for Apple Silicon / Linux
    train_loader = DataLoader(train_set, batch_size=batch_size, 
                              shuffle=True, num_workers=2)
    
    test_loader = DataLoader(test_set, batch_size=batch_size, 
                             shuffle=False, num_workers=2)

    return train_loader, test_loader

# Optional: verify data shape when running this file directly
if __name__ == '__main__':
    train_loader, _ = get_loaders()
    images, labels = next(iter(train_loader))
    print(f"Batch Shape: {images.shape}")
    print(f"Labels Shape: {labels.shape}")
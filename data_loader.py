import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

def get_loaders(batch_size=128, use_subset=False):
    """
    Returns train_loader (with Augmentation) and test_loader (Clean).
    """
    
    # Standard CIFAR-10 stats
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    # --- CHANGE 1: Training Transform (Augmentation) ---
    train_transform = transforms.Compose([
        # Randomly crop a 32x32 piece from the image (with 4px padding)
        # This forces the model to recognize the object even if it's off-center
        transforms.RandomCrop(32, padding=4),
        
        # Randomly flip the image horizontally (50% chance)
        # Teaches: A car pointing left is still a car
        transforms.RandomHorizontalFlip(),
        
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    # --- CHANGE 2: Test Transform (No Augmentation) ---
    # We only normalize the test set. No randomness allowed here.
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    # Apply specific transforms to specific sets
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=train_transform)
    
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=test_transform)

    # Subset Logic
    if use_subset:
        indices = range(10000)
        train_set = Subset(train_set, indices)
        print("Using Subset: 10,000 images (Augmented)")
    else:
        print("Using Full Dataset: 50,000 images (Augmented)")

    # Create Loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, 
                              shuffle=True, num_workers=2)
    
    test_loader = DataLoader(test_set, batch_size=batch_size, 
                             shuffle=False, num_workers=2)

    return train_loader, test_loader
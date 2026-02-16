import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from model import ImprovedCNN  
from data_loader import get_loaders

def main():
    # 1. Setup Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Get Data (Now includes Augmentation automatically)
    trainloader, testloader = get_loaders(batch_size=128, use_subset=False)

    # 3. Initialize Model
    model = ImprovedCNN().to(device)

    # 4. Define Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    # We stick to the same LR (0.001) for a fair comparison
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. Training Loop
    EPOCHS = 25
    train_losses = []
    test_accs = []
    
    print(f"Starting training for {EPOCHS} epochs (Improved Model)...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train() 
        running_loss = 0.0
        
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Stats
        avg_loss = running_loss / len(trainloader)
        train_losses.append(avg_loss)
        
        # Evaluate
        acc = evaluate(model, testloader, device)
        test_accs.append(acc)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}%")

    print(f"Finished in {time.time() - start_time:.1f} seconds")
    
    # Save the Improved Model
    torch.save(model.state_dict(), 'improved_cnn.pth') 
    print("Model saved to improved_cnn.pth")
    
    plot_results(train_losses, test_accs)

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def plot_results(losses, accs):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss', color=color)
    ax1.plot(losses, color=color, marker='o')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx() 
    color = 'tab:blue'
    ax2.set_ylabel('Test Accuracy (%)', color=color) 
    ax2.plot(accs, color=color, marker='x')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Training Dynamics (Improved CNN)')
    plt.show()

if __name__ == '__main__':
    main()
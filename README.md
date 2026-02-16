# Project Report: CIFAR-10 Classification Learning

## Phase 1: The Baseline Model (Simple CNN)

### 1. Setup

The initial test used a `BasicCNN` to see how a standard setup performs without any extra help.

* **Structure:** 2 Convolutional blocks, followed by 2 Linear layers.
* **Size:** About 1.6 million parameters.
* **Settings:** No Dropout, no Batch Normalization.

### 2. The Problem: "Memorization" Overfitting

Training for 10 epochs showed a clear issue.

| Metric | Start | End | What happened |
| --- | --- | --- | --- |
| **Training Loss** | 1.32 | **0.06** | The model learned the training data perfectly. |
| **Test Accuracy** | 63% | **73%** | The model struggled with new images. |

**The Gap:**
The model achieved nearly 100% accuracy on images it had seen (Training) but stuck at 73% on images it hadn't (Test). This huge gap means the network didn't learn what a "bird" looks like; it just memorized the specific pixels of the birds in the training folder.

### 3. Why It Failed

* **Neurons depended on each other:** Without **Dropout**, neurons didn't learn independent features. If one neuron made a mistake, another just adjusted to cover for it, rather than learning a real pattern.
* **Unstable Layers:** Without **Batch Normalization**, deep layers had to constantly adjust to changing inputs from previous layers, making learning slow and shaky.
* **Static Data:** The model saw the exact same 50,000 images every time. It learned that a car is only a car if it's facing left, because that's all it saw.

---

## Phase 2: The Improved Model (Robust CNN)

### 1. The Fixes

To stop the memorization, the `ImprovedCNN` introduced three specific constraints to make training "harder" but more effective.

* **More Depth:** Added a 3rd Convolutional block (128 channels) to understand complex shapes.
* **Batch Normalization:** Added after every convolution to stabilize the math inside the network.
* **Dropout (0.5):** Randomly turned off 50% of the neurons during training to force independence.
* **Data Augmentation:** Randomly flipped and cropped images so the model never saw the exact same picture twice.

### 2. Results Comparison

| Metric | Baseline | Improved | Difference |
| --- | --- | --- | --- |
| **Training Loss** | 0.06 (Easy) | **0.55 (Hard)** | The model had to work harder. |
| **Test Accuracy** | 73% | **82%** | Much better at real-world tasks. |
| **Gap** | 25% | **3%** | Overfitting solved. |

### 3. Why It Worked

#### A. Data Augmentation (Forcing Invariance)

By randomly flipping images left and right, the model couldn't rely on simple pixel positions.

* **Mechanism:** If a car faces left in one epoch and right in the next, the model is forced to learn the *shape* of a car, not just its location on the screen.

#### B. Dropout (Forcing Independence)

Dropout randomly disables half the network during every training step.

* **Mechanism:** No single neuron can be the "hero." The network has to build multiple, redundant pathways to identify features (like wings or wheels) because it never knows which path will be broken. This creates a more robust "consensus" prediction.

#### C. Batch Normalization (Enabling Depth)

This step standardizes the inputs between layers.

* **Mechanism:** It keeps the data centered (mean of 0, variance of 1) as it flows through the network. This prevents the "vanishing gradient" problem, allowing the deeper 3-block architecture to train successfully without getting stuck.

### 4. Conclusion

The baseline model proved that raw power isn't enough; it just memorized the answers. By adding constraints (Dropout, Augmentation), the training process became more difficult (higher loss), but the resulting model actually *learned* the features, leading to an 82% accuracy on the test set. Robustness matters more than low training loss.

---

# Stage 1 Analysis: The Baseline CNN

### 1. Model Architecture Overview

The baseline model (`BasicCNN`) was a standard, shallow Convolutional Neural Network designed to establish a performance floor.

* **Depth:** 2 Convolutional Blocks + 2 Fully Connected Layers.
* **Parameters:** ~1.6 Million (dominated by the first Dense layer).
* **Regularization:** None (No Dropout, No Batch Normalization, No Weight Decay).
* **Optimizer:** Adam (`lr=0.001`).

### 2. Performance Metrics

The training run over 10 epochs yielded the following critical data points:

| Metric | Start (Epoch 1) | End (Epoch 10) | Trend |
| --- | --- | --- | --- |
| **Training Loss** | 1.32 | **0.062** |  Rapid Decrease (Near Zero) |
| **Test Accuracy** | 63% | **73.5%** |  Plateaued after Epoch 4 |

### 3. Critical Failure: The Generalization Gap

The most significant finding is the massive divergence between Training performance and Test performance.

* **Training Accuracy (Estimated):** A loss of `0.06` corresponds to roughly **98-99% accuracy** on the training set.
* **Test Accuracy:** Stalled at **73.5%**.
* **The Gap:** There is a **~25% gap** between what the model knows (Training) and what it can predict (Test).

#### Diagnosis: Severe Overfitting

The model has "memorized" the training data rather than learning generalizable features.

1. **Memorization:** The model learned the specific noise and pixel arrangements of the 50,000 training images.
2. **Lack of Robustness:** When presented with a *new* image (Test set) that varies slightly in lighting or orientation, the model fails because it learned specific examples, not general rules.

### 4. Root Cause Analysis

Why did this architecture overfit so heavily?

#### A. Missing Regularization (Dropout)

The fully connected layer (`self.fc1`) has **~2 million connections** (). Without **Dropout**, these neurons co-adapted to fix errors for specific training images.

* *Theory:* If Neuron A learns "Pointy Ears", Neuron B should learn "Whiskers".
* *Reality:* Without Dropout, Neuron B might just learn "The specific background pixel in Image #402 is blue," which helps minimize training loss but is useless for testing.

#### B. Internal Covariate Shift (Missing Batch Norm)

As the network trains, the distribution of inputs to inner layers changes constantly (parameters in `conv1` change  input to `conv2` changes).

* **Consequence:** The later layers have to constantly "chase" the moving target of the previous layers.
* **Result:** The model is unstable and highly sensitive to initialization. It requires a lower learning rate to converge safely, which slows down training.

#### C. Static Data (No Augmentation)

The model saw the exact same 50,000 images in the exact same orientation every epoch.

* **The Flaw:** If a "Plane" in the training set always points right, the model learns "Right-pointing blob = Plane".
* **The Reality:** If the Test set has a plane pointing left, the model fails.

### 5. Conclusion & Next Steps

The `BasicCNN` successfully proved that the architecture *can* learn (going from 10% random guess to 73%). However, it has hit a "capacity ceiling" due to overfitting. Adding more layers now would likely *hurt* performance (making it memorize even faster).

**Stage 2 Objectives:**
To bridge the 25% gap, we must introduce **Regularization**:

1. **Add Batch Normalization:** To stabilize layer inputs and allow faster training.
2. **Add Dropout (0.5):** To randomy disable neurons during training, forcing the network to learn redundant, robust features.
3. **Data Augmentation:** To artificially expand the dataset by flipping and shifting images.

---

### Visuals for your Report

To make this section "pop" in your document, you should generate these two plots using `matplotlib`:

1. **Loss Curve:** Show Training Loss plummeting to 0 while Test Loss (if you tracked it) starts rising again.
2. **Confusion Matrix:** Show *which* classes are being confused (e.g., Cat vs Dog is a common error in CIFAR-10).
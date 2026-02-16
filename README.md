

---

# Experimental Report: CIFAR-10 Classification with PyTorch

## Part 1: Baseline Model (Stage 1)

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

The most significant finding was the massive divergence between Training performance and Test performance.

* **Training Accuracy (Estimated):** A loss of `0.06` corresponds to roughly **98-99% accuracy** on the training set.
* **Test Accuracy:** Stalled at **73.5%**.
* **The Gap:** There is a **~25% gap** between what the model knows (Training) and what it can predict (Test).

#### Diagnosis: Severe Overfitting

The model "memorized" the training data rather than learning generalizable features.

1. **Memorization:** The model learned the specific noise and pixel arrangements of the 50,000 training images.
2. **Lack of Robustness:** When presented with a *new* image (Test set) that varied slightly in lighting or orientation, the model failed because it learned specific examples, not general rules.

### 4. Root Cause Analysis

Why did this architecture overfit so heavily?

#### A. Missing Regularization (Dropout)

The fully connected layer (`self.fc1`) has **~2 million connections** (). Without **Dropout**, these neurons co-adapted to fix errors for specific training images.

* *Theory:* If Neuron A learns "Pointy Ears", Neuron B should learn "Whiskers".
* *Reality:* Without Dropout, Neuron B might just learn "The specific background pixel in Image #402 is blue," which helps minimize training loss but is useless for testing.

#### B. Internal Covariate Shift (Missing Batch Norm)

As the network trains, the distribution of inputs to inner layers changes constantly.

* **Consequence:** The later layers have to constantly "chase" the moving target of the previous layers.
* **Result:** The model is unstable and highly sensitive to initialization.

#### C. Static Data (No Augmentation)

The model saw the exact same 50,000 images in the exact same orientation every epoch.

* **The Flaw:** If a "Plane" in the training set always points right, the model learns "Right-pointing blob = Plane".
* **The Reality:** If the Test set has a plane pointing left, the model fails.

---

## Part 2: Improved Model (Stage 2)

### 1. Model Architecture Overview

To address the severe overfitting observed in Stage 1, we deployed the `ImprovedCNN` architecture. This model was designed not just for capacity, but for **robustness**.

* **Depth:** Increased to 3 Convolutional Blocks (64  128 channels).
* **Regularization:** Added **Batch Normalization** (after every Conv) and **Dropout** (0.5 before Classifier).
* **Data Pipeline:** Transformed the static dataset into a dynamic one using **Data Augmentation** (Random Crop + Horizontal Flip).

### 2. Performance Comparison (Stage 1 vs. Stage 2)

| Metric | Stage 1 (Baseline) | Stage 2 (Improved) | Change |
| --- | --- | --- | --- |
| **Training Loss** | 0.062 (Near Zero) | **~0.55** |  Higher (Harder task) |
| **Test Accuracy** | 73.56% | **~82.0%** |  Significant Improvement |
| **Generalization Gap** | ~25% (Huge) | **~3% (Healthy)** |  Gap Closed |

### 3. Critical Success: Closing the Gap

The most important result of Stage 2 is not just the higher accuracy, but the **convergence behavior**.

* **Training Loss Increased:** Interestingly, the final training loss is *higher* than in Stage 1 (0.55 vs 0.06). This is expected and desirable. By adding Dropout and Augmentation, we made the training task harder. The model could no longer "cheat" by memorizing pixels; it had to struggle to learn real features.
* **Test Accuracy Increased:** Despite the harder training task, the model performed much better on unseen data.
* **The Conclusion:** The model has shifted from **Memorization** (Stage 1) to **Generalization** (Stage 2).

### 4. Why It Worked (Mechanism of Action)

#### A. Data Augmentation  Invariance

In Stage 1, the model failed if a "Car" was facing the wrong way.

* **The Fix:** By randomly flipping and cropping images during training, we forced the model to learn **Translation and Orientation Invariance**.
* **Result:** The model now recognizes a car regardless of whether it is centered or facing left/right.

#### B. Dropout  Feature Robustness

In Stage 1, neurons co-adapted to fix each other's errors.

* **The Fix:** `Dropout(0.5)` randomly silenced 50% of the neurons at every step.
* **Result:** No single neuron could be relied upon. The network was forced to learn redundant, distributed representations of features (e.g., multiple neurons distinctively recognizing "wheels").

#### C. Batch Normalization  Training Stability

In Stage 1, we were limited to 2 layers because deeper networks are hard to train.

* **The Fix:** Batch Norm standardized the inputs to every layer ().
* **Result:** This allowed us to successfully add a **3rd Convolutional Block** (increasing depth to 128 channels) without the gradients vanishing or exploding, giving the model the capacity to understand more complex shapes.

### 5. Final Conclusion

The experiment successfully diagnosed a high-variance (overfitting) problem in the baseline CNN. By introducing regularization (Dropout, BatchNorm) and data augmentation, we reduced the generalization gap from 25% to <5% and increased final accuracy from 73% to 82%. This confirms that for small datasets like CIFAR-10, model robustness is as critical as model capacity.
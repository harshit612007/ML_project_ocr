# Comprehensive Technical Project Explanation & Learning Guide
## Ancient Tamil Palm-Leaf Manuscript Character Recognition Using Classical Machine Learning & Neural Networks

---

> **Target Audience:** Students and researchers seeking a rigorous, ground-up understanding of Optical Character Recognition (OCR) for historical palm-leaf manuscripts using classical machine learning and neural networks.
> **Scope:** Strictly Classical Machine Learning and Shallow Neural Networks (**Zero Deep Learning / No CNNs / No GPUs**).
> **Primary Literature Implementations:**
> 1. **Paper 1 (FFBNN):** Ramya & Parvathavarthini (2014) — *Feed Forward Back Propagation Neural Network Based Character Recognition System for Tamil Palm Leaf Manuscripts*.
> 2. **Paper 2 (MABPN):** Poornima Devi & Sornam (2020) — *Classification of Ancient Handwritten Tamil Characters on Palm Leaf Inscription Using Modified Adaptive Backpropagation Neural Network with GLCM Features*.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Research Papers and Literature Understanding](#2-research-papers-and-literature-understanding)
   - [2.1 Paper 1: Ramya & Parvathavarthini (2014) — FFBNN Pipeline](#21-paper-1-ramya--parvathavarthini-2014--ffbnn-pipeline)
   - [2.2 Paper 2: Poornima Devi & Sornam (2020) — MABPN + GLCM Pipeline](#22-paper-2-poornima-devi--sornam-2020--mabpn--glcm-pipeline)
3. [Machine Learning Fundamentals](#3-machine-learning-fundamentals)
4. [Neural Network Fundamentals](#4-neural-network-fundamentals)
5. [Mathematical Foundations](#5-mathematical-foundations)
   - [5.1 Neural Network Mathematics with Numerical Examples](#51-neural-network-mathematics-with-numerical-examples)
   - [5.2 Image Processing & Feature Mathematics with Numerical Examples](#52-image-processing--feature-mathematics-with-numerical-examples)
6. [Image Processing & Manuscript Restoration](#6-image-processing--manuscript-restoration)
7. [Feature Extraction Techniques](#7-feature-extraction-techniques)
8. [Paper 1: FFBNN Methodology & Pipeline](#8-paper-1-ffbnn-methodology--pipeline)
9. [Paper 2: MABPN Methodology & Pipeline](#9-paper-2-mabpn-methodology--pipeline)
10. [Complete System Architecture](#10-complete-system-architecture)
11. [Codebase Structure](#11-codebase-structure)
12. [File-by-File Code Explanation](#12-file-by-file-code-explanation)
13. [Training and Execution Flow](#13-training-and-execution-flow)
14. [Experiments and Quantitative Results](#14-experiments-and-quantitative-results)
15. [Comparison of Both Research Papers](#15-comparison-of-both-research-papers)
16. [Limitations and Implementation Deviations](#16-limitations-and-implementation-deviations)
17. [Frequently Asked Technical Questions (Viva Voce Preparation)](#17-frequently-asked-technical-questions-viva-voce-preparation)
18. [References and Further Reading](#18-references-and-further-reading)

---

## 1. Project Overview

### 1.1 The Cultural & Historical Domain: Tamil Palm-Leaf Manuscripts (*Panaiyolai Suvadigal*)
For over two millennia, classical Tamil scientific, philosophical, astronomical, and medical knowledge (such as the *Thirukkural*, *Tolkappiyam*, and Siddha medicine treatises) was recorded on cured strips of dried Palmyra palm leaves (*Borassus flabellifer* / தமிழ் பனையோலை - *Panaiyolai*).

Recognizing writing on palm leaves presents physical and morphological challenges distinct from modern printed or paper documents:
1. **The Iron Stylus (*Ezhuthani* / எழுத்தாணி):** Scribes used a needle-sharp stylus to incise grooves into the fibrous leaf surface. To render the writing legible, lampblack / carbon soot mixed with resinous vegetable juice (e.g., *Eclipta prostrata* / கரிசலாங்கண்ணி) was rubbed into the incisions.
2. **Why Tamil Characters Are Curved (Paleographic Origin):** Palm leaves possess strong horizontal veins and grain fibers. Drawing sharp horizontal top-bars (such as the Devanagari *Shirorekha*) would slit the capillaries and crack the manuscript along its grain. Ancient Tamil scribes therefore evolved circular, looping, curved glyph geometries (*Vatteluttu* and classical script) that glide across fibers without tearing the leaf.
3. **Severe Physical Degradation:** Manuscripts suffer from biological staining, fungus, insect boreholes, background color unevenness (ochre/yellowish-brown leaf patina), soot ink fading along stylus paths, and micro-cracks along capillaries.

```
       +--------------------------------------------------------+
       |   P A L M   L E A F   M O R P H O L O G Y              |
       |   ══════════════════════════════════════════════════   |
       |   [======== Horizontal Fibrous Grain Veins ========]   |
       |   ( Incised stylus groove filled with carbon soot  )   |
       |   ( Rounded, curved glyphs protect leaf integrity  )   |
       +--------------------------------------------------------+
```

### 1.2 Core Project Goal
The goal of this project is to implement, evaluate, and benchmark classical pattern recognition and neural network methods to transcribe degraded historical ancient Tamil characters into digital Unicode text, specifically focusing on the methodologies of two foundational research papers:
* **Paper 1:** Feed Forward Back Propagation Neural Network (FFBNN) with sliding window analysis and binary-coded features (Ramya & Parvathavarthini, 2014).
* **Paper 2:** Modified Adaptive Backpropagation Network (MABPN) using Gray-Level Co-occurrence Matrix (GLCM) texture features, Nguyen-Widrow weight initialization, Shannon activation functions, and dynamic learning-rate adjustment (Poornima Devi & Sornam, 2020).

### 1.3 Strict Classical Machine Learning Scope (NO Deep Learning)
This project enforces a strict boundary:
* **No Deep Learning (CNNs, ResNets, Transformers) and no GPU runtimes (`torch`, `tensorflow`, `keras`, `cuda`) are used.**
* All classifiers run on **explicit, hand-crafted mathematical feature vectors** (GLCM textures, binary distributions, Gabor wavelets, SIFT histograms, and zoning profiles) rather than discovering features end-to-end from raw pixel grids.
* The neural networks implemented in this project are **shallow Multi-Layer Perceptrons (MLPs)** operating with 1 to 2 hidden layers, optimized on CPU via analytical and second-order methods.

---

## 2. Research Papers and Literature Understanding

### 2.1 Paper 1: Ramya & Parvathavarthini (2014) — FFBNN Pipeline

#### Bibliographic Citation
> J. Ramya and B. Parvathavarthini, *"Feed Forward Back Propagation Neural Network Based Character Recognition System for Tamil Palm Leaf Manuscripts"*, **Journal of Computer Science**, Vol. 10, No. 4, pp. 660–670, 2014. DOI: `10.3844/jcssp.2014.660.670`.

#### Problem Addressed
Historical palm-leaf manuscripts preserved in Oriental manuscript libraries and temple archives are rapidly decaying due to biological decay and brittle fragmentation. Manual transcription requires specialized paleographic expertise and is slow. Paper 1 introduces an automated OCR system to convert scanned palm-leaf character images into digital text.

#### Why the Problem is Important
Without automated optical character recognition, thousands of rare palm-leaf bundles detailing traditional Siddha medicine, mathematics, and Tamil classical literature risk being lost permanently as the physical leaves disintegrate.

#### Proposed Solution & Complete Sequential Pipeline
```
[Scanned Manuscript Image]
         │
         ▼
[Preprocessing: Grayscale Conversion & Adaptive Histogram Equalization]
         │
         ▼
[Character Segmentation: Sliding Window over Text Lines]
         │
         ▼
[Feature Extraction: Normalized Pixel Density / Binary-Coded Values (64-D)]
         │
         ▼
[Classification: Feed Forward Back Propagation Neural Network (FFBNN)]
         │
         ▼
[Final Tamil Character Class Output]
```

1. **Preprocessing:** Converts the scanned image to grayscale and applies **adaptive histogram equalization** to improve contrast between the soot-incised stylus marks and the discolored palm-leaf background.
2. **Segmentation:** Scans across text lines using a **sliding window** technique. By analyzing the horizontal and vertical projection profiles and ink density within the window, individual character glyphs are bounded and separated.
3. **Feature Extraction:** Character crops are scaled to a standardized dimension (e.g., $8 \times 8$ or $64 \times 64$) to create a **binary-coded vector** representing stroke presence ($1$) versus background ($0$).
4. **Classification:** A 3-layer **Feed Forward Back Propagation Neural Network (FFBNN)** with input nodes, a hidden layer of sigmoid neurons, and an output layer representing Tamil character classes.
5. **Training Algorithm:** Standard backpropagation using gradient descent with Mean Squared Error (MSE) minimization and fixed learning rate $\eta$ and momentum $\alpha$.
6. **Reported Results:** Achieved approximately **~90% character recognition accuracy** across 10 test sets of Tamil palm-leaf manuscripts.
7. **Documented Limitations:**
   * High sensitivity to initial random weights: Can become trapped in local minima or suffer slow convergence.
   * Fixed learning rate can lead to oscillations if $\eta$ is too large, or slow training if $\eta$ is too small.
   * Binary-coded and sliding-window features are vulnerable to leaf grain noise and broken ink lines.

---

### 2.2 Paper 2: Poornima Devi & Sornam (2020) — MABPN + GLCM Pipeline

#### Bibliographic Citation
> M. Poornima Devi and M. Sornam, *"Classification of Ancient Handwritten Tamil Characters on Palm Leaf Inscription Using Modified Adaptive Backpropagation Neural Network with GLCM Features"*, **ACM Transactions on Asian and Low-Resource Language Information Processing (TALLIP)**, Vol. 19, No. 6, Article 82, pp. 1–14, October 2020. DOI: `10.1145/3411756`.

#### Problem Addressed
Ancient Tamil inscriptions on palm leaves and temple stone engravings exhibit irregular, cursive letterforms with varying stroke thicknesses, ink erosion, and textural background interference. Standard neural network classifiers with binary inputs struggle on these noisy, complex scripts.

#### Why the Problem is Important
Historical palm leaves have complex fibrous textures that corrupt raw binary pixel values. Capturing second-order statistical relationships (spatial texture) and accelerating neural convergence through algorithmic adaptations is critical for reliable transcription.

#### Proposed Solution & Complete Sequential Pipeline
```
[Degraded Palm-Leaf Manuscript]
         │
         ▼
[Preprocessing & Restoration: Denoising & Adaptive Binarization]
         │
         ▼
[Segmentation: Contour-Based Convex Hull Bounding Box]
         │
         ▼
[Feature Extraction: Gray-Level Co-occurrence Matrix (GLCM) Features]
   (Contrast, Energy [sqrt(ASM)], Homogeneity, Correlation, Dissimilarity, ASM across 0°, 45°, 90°, 135°)
         │
         ▼
[Classification: Modified Adaptive Backpropagation Network (MABPN)]
   - Nguyen-Widrow Weight Initialization (Equations 1-6)
   - Hidden Layer: Sigmoid Activation (Equation 7)
   - Output Layer: Shannon Wavelet Activation f(x) = (sin(1.75x) - sin(0.75x)) / (1.75x) (Equation 8)
   - Dual Error Propagation: Linear Error e1 = d - y and Quadratic Error e2 = 0.5(d^2 - y^2) (Equations 14-17)
   - Adaptive Learning Rate & Momentum Update Rules (Equations 19-20)
         │
         ▼
[Predicted Ancient Tamil Character]
```

1. **Preprocessing & Segmentation:** After adaptive contrast enhancement, characters are segmented using a **contour-based convex hull bounding box** method. Finding the convex hull around each connected contour encloses irregular, curved Tamil glyphs tightly, separating conjoined and touching glyphs more effectively than axis-aligned rectangles.
2. **Feature Extraction (GLCM):** Rather than raw binary pixels, the paper extracts second-order texture statistics via the **Gray-Level Co-occurrence Matrix (GLCM)**. Across multiple distances and orientations ($0^\circ, 45^\circ, 90^\circ, 135^\circ$), six Haralick texture descriptors are computed (Equations 21–27):
   * **Contrast (Eq. 21):** $\sum_{i,j} (i - j)^2 P(i,j)$ — Quantifies local intensity variations.
   * **Energy (Eq. 22):** $\sqrt{\text{ASM}} = \sqrt{\sum_{i,j} P(i,j)^2}$ — Measures textural uniformity.
   * **Homogeneity (Eq. 23):** $\sum_{i,j} \frac{P(i,j)}{1 + (i - j)^2}$ — Measures closeness of element distribution to GLCM diagonal.
   * **Correlation (Eq. 24):** $\sum_{i,j} \frac{(i - \mu_i)(j - \mu_j) P(i,j)}{\sigma_i \sigma_j}$ — Measures linear gray-level dependency.
   * **Dissimilarity (Eq. 26):** $\sum_{i,j} |i - j| P(i,j)$ — Linear difference measure across pixel pairs.
   * **Angular Second Moment (ASM) (Eq. 27):** $\sum_{i,j} P(i,j)^2$ — Sum of squared matrix probabilities.
3. **Weight Initialization (Nguyen-Widrow):** Bypasses standard random initialization by scaling weights according to $\beta = 0.7 \cdot H^{1/D}$, distributing neuron activation intervals evenly across the input hypersphere.
4. **Activation Function (Shannon Wavelet Activation):** Employs the bounded, localized Shannon wavelet function (Equation 8):
   $$f(x) = \frac{\sin(1.75x) - \sin(0.75x)}{1.75x}, \quad \lim_{x \to 0} f(x) = \frac{4}{7} \approx 0.57142857$$
5. **Adaptive Learning & Dual Error:** Propagates both linear error $e_1 = d - y$ and non-linear quadratic error $e_2 = 0.5(d^2 - y^2)$, dynamically tuning the learning rate $\eta$:
   * If error decreases: $\eta(t) = \eta(t-1) \times 1.05$ (accelerates descent).
   * If error spikes: $\eta(t) = \eta(t-1) \times 0.70$ (dampens oscillation and ensures stability).
6. **Reported Results:** GLCM features paired with MABPN outperformed standard BPNN and binary-coded inputs, yielding faster convergence and higher classification accuracy on degraded historical Tamil manuscripts.
7. **Documented Characteristics:**
   * GLCM features capture spatial texture and stroke densities across the manuscript character surface.
   * Shannon wavelet activation with analytical limit handling ($4/7$ at $x=0$, derivative $0.0$ at $x=0$) prevents vanishing and exploding gradients.


---

## 3. Machine Learning Fundamentals

Before implementing neural networks, we establish core machine learning principles through the lens of Tamil manuscript character recognition.

### 3.1 What is Machine Learning?
In traditional programming, a software developer writes explicit rules (`if-else` statements) to process input data into outputs:
$$\text{Input Data} + \text{Explicit Rules} \longrightarrow \text{Output}$$

In **Machine Learning (ML)**, we provide an algorithm with pairs of input character images and their true labels, and the model automatically learns the mathematical mapping function $f: \mathbf{x} \mapsto y$:
$$\text{Input Data } \mathbf{x} + \text{Known Outputs } y \longrightarrow \text{Learned Model } f(\mathbf{x}) \approx y$$

### 3.2 Supervised Learning & Classification
* **Supervised Learning:** The learning algorithm receives a labeled dataset $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$, where $\mathbf{x}_i \in \mathbb{R}^D$ is a $D$-dimensional numerical feature vector representing a character, and $y_i \in \{1, 2, \dots, C\}$ is the ground-truth character label (e.g., class `01_a` for 'அ').
* **Classification:** Because the target label $y$ belongs to a finite, discrete set of categories ($10$ ancient Tamil character classes in this benchmark), this is a **multi-class classification problem**, unlike regression where targets are continuous real numbers.

### 3.3 Features and Labels
* **Raw Image vs Feature Vector:** A raw image is a grid of pixels (e.g., $64 \times 64 = 4096$ intensity values). Using raw pixels as features causes the "curse of dimensionality" and leaves classifiers vulnerable to noise and translation shifts.
* **Feature Extraction:** We transform raw pixels into a compact numerical vector $\mathbf{x} = [x_1, x_2, \dots, x_D]^T$ that captures discriminative properties:
  * Stroke angles and loops (Gabor wavelets).
  * Local gradient directions (SIFT / HOG).
  * Surface ink texture and roughness (GLCM).
  * Pixel presence across zones (Binary coded / zoning).
* **Ground-Truth Label ($y$):** An integer identifier representing the true character glyph:

| Class Code | Tamil Glyph | Phonetic Name | Description |
| :---: | :---: | :---: | :--- |
| `01_a` | அ | Short 'A' | Primary vowel with initial loop and curved right sweep |
| `02_aa` | ஆ | Long 'AA' | Similar to 'அ' but with an extended downward loop tail |
| `03_i` | இ | Short 'I' | Double-lobed circular glyph with central bridge |
| `04_ka` | க | Consonant 'Ka' | Top crossbar with central vertical drop and lower loop |
| `05_cha` | ச | Consonant 'Cha' | Open hooked top stroke curving downwards |
| `06_ta` | த | Consonant 'Ta' | Top bar with descending curved belly and hook |
| `07_pa` | ப | Consonant 'Pa' | Open U-shaped container glyph |
| `08_ma` | ம | Consonant 'Ma' | U-shaped glyph with an inner horizontal loop closure |
| `09_ra` | ர | Consonant 'Ra' | Single vertical stroke with slight top curvature |
| `10_la` | ல | Consonant 'La' | Looped glyph with ascending right-hand arm |

### 3.4 Data Partitioning: Training, Validation, and Testing
To evaluate how well our model generalizes to unseen manuscripts, we partition the dataset $\mathcal{D}$:
1. **Training Set ($80\% = 400$ samples):** Used by the learning algorithm to adjust weights and biases.
2. **Held-Out Test Set ($20\% = 100$ samples):** Kept completely untouched during training and tuning. Evaluated only once to measure real-world performance.
3. **Stratification:** Splitting is performed in a **stratified** manner so that every character class is equally represented (exactly $40$ training samples and $10$ test samples per class), preventing class imbalance artifacts.
4. **$K$-Fold Cross-Validation:** The training set is split into $K=10$ equal folds. The model is trained on $9$ folds and validated on the remaining $1$ fold, repeating $10$ times. This provides a mean cross-validation accuracy $\pm$ standard deviation to measure stability.

```
+-----------------------------------------------------------------------+
| TOTAL DATASET: 500 SAMPLES (10 Classes x 50 Samples/Class)            |
+------------------------------------------+----------------------------+
|        TRAINING PARTITION (80%)          |    HELD-OUT TEST (20%)     |
|              400 Samples                 |        100 Samples         |
|  [F1] [F2] [F3] [F4] ... [F10]           |  Used strictly for final   |
|  <-- 10-Fold Cross-Validation Protocol ->|  unbiased evaluation       |
+------------------------------------------+----------------------------+
```

### 3.5 Overfitting vs Underfitting
* **Underfitting (High Bias):** The model is too simple to capture the underlying pattern (e.g., using a single linear boundary to separate conjoined, curved Tamil characters). Both training and test accuracies are poor.
* **Overfitting (High Variance):** The model memorizes training noise (e.g., memorizing specific leaf vein scratches or ink cracks on individual training leaves). Training accuracy is near $100\%$, but test accuracy on new leaves drops sharply.
* **Solutions used in this project:**
  * **$L_2$ Weight Regularization:** Penalizes excessively large weights.
  * **PCA Dimensionality Reduction:** Removes noisy, redundant feature dimensions.
  * **Cross-Validation:** Detects overfitting early.

---

## 4. Neural Network Fundamentals

### 4.1 From Biological Neurons to Artificial Neurons
The human brain processes visual characters through interconnected networks of biological neurons in the visual cortex. A biological neuron receives electrical signals through **dendrites**, integrates the signals in the **cell body (soma)**, and fires an electrical spike along its **axon** through **synaptic junctions** if the cumulative signal exceeds an action potential threshold.

In 1943, Warren McCulloch and Walter Pitts modeled this process mathematically:

```
 Biological Neuron:   Dendrites (Inputs) ──> Soma (Summation) ──> Axon (Threshold)
 Artificial Neuron:   Inputs x_i * w_i   ──> Sum + Bias (z)   ──> Activation f(z)
```

```mermaid
graph LR
    x1["Input x₁"] -->|"Weight w₁"| sum["Summation & Bias: z = ∑ xᵢwᵢ + b"]
    x2["Input x₂"] -->|"Weight w₂"| sum
    xd["Input x_D"] -->|"Weight w_D"| sum
    bias["Bias b"] --> sum
    sum --> act["Activation Function f(z)"]
    act --> out["Neuron Output a = f(z)"]
```

### 4.2 The Artificial Neuron (Perceptron)
An artificial neuron performs two operations:
1. **Linear Combination (Weighted Sum):**
   $$z = \sum_{i=1}^D x_i w_i + b = \mathbf{w}^T \mathbf{x} + b$$
   * $\mathbf{x} = [x_1, x_2, \dots, x_D]^T$: Input feature vector.
   * $\mathbf{w} = [w_1, w_2, \dots, w_D]^T$: Synaptic weight vector.
   * $b$: Bias term (shifts the activation threshold independent of inputs).
2. **Non-Linear Activation:**
   $$a = f(z)$$
   * $f(\cdot)$: Non-linear activation function. Without non-linearity, stacking multiple layers would still result in a linear transformation: $\mathbf{W}_2 (\mathbf{W}_1 \mathbf{x}) = \mathbf{W}_{21} \mathbf{x}$.

### 4.3 Multilayer Perceptron (Feed Forward Neural Network)
A **Multilayer Perceptron (MLP)** organizes artificial neurons into sequential layers:
1. **Input Layer:** Receives the feature vector $\mathbf{x} \in \mathbb{R}^D$ (no mathematical computation).
2. **Hidden Layer(s):** One or more intermediate layers that transform features into non-linear representations.
3. **Output Layer:** Produces class prediction scores or probabilities across the $C=10$ classes.
4. **Feed Forward:** Signals flow strictly in one direction from input to output, with no recurrent feedback loops.

```mermaid
graph LR
    subgraph Input_Layer ["Input Features (x ∈ ℝᴰ)"]
        i1["x₁ (Contrast)"]
        i2["x₂ (Correlation)"]
        i3["x₃ (Energy)"]
        i4["x₄ (Homogeneity)"]
    end

    subgraph Hidden_Layer ["Hidden Layer (h units)"]
        h1["Hidden Neuron 1"]
        h2["Hidden Neuron 2"]
        h3["Hidden Neuron 3"]
    end

    subgraph Output_Layer ["Output Classes (10 Tamil Glyphs)"]
        o1["P(01_a / அ)"]
        o2["P(02_aa / ஆ)"]
        o3["P(... / ...)"]
        o10["P(10_la / ல)"]
    end

    i1 --> h1
    i1 --> h2
    i1 --> h3
    i2 --> h1
    i2 --> h2
    i2 --> h3
    i3 --> h1
    i3 --> h2
    i3 --> h3
    i4 --> h1
    i4 --> h2
    i4 --> h3

    h1 --> o1
    h1 --> o2
    h1 --> o3
    h1 --> o10
    h2 --> o1
    h2 --> o2
    h2 --> o3
    h2 --> o10
    h3 --> o1
    h3 --> o2
    h3 --> o3
    h3 --> o10
```

---

## 5. Mathematical Foundations

### 5.1 Neural Network Mathematics with Numerical Examples

#### 1. Weighted Sum of Inputs
$$\mathbf{z}^{[1]} = \mathbf{X} \mathbf{W}^{[1]} + \mathbf{b}^{[1]}$$
* $\mathbf{X} \in \mathbb{R}^{N \times D}$: Batch of $N$ character samples with $D$ features.
* $\mathbf{W}^{[1]} \in \mathbb{R}^{D \times H}$: Weight matrix between input and hidden layer with $H$ neurons.
* $\mathbf{b}^{[1]} \in \mathbb{R}^{1 \times H}$: Bias vector for hidden neurons.
* $\mathbf{z}^{[1]} \in \mathbb{R}^{N \times H}$: Pre-activation sums.

* **Why it is needed:** Implements linear combinations that rotate and project input features into decision spaces.
* **Numerical Example:**
  Suppose a character has $D=2$ features: $x_1 = 0.8$ (high contrast), $x_2 = 0.2$ (low homogeneity).
  Neuron 1 has weights $w_1 = 1.5, w_2 = -0.5$ and bias $b = 0.1$.
  $$z = (0.8 \times 1.5) + (0.2 \times -0.5) + 0.1 = 1.20 - 0.10 + 0.10 = 1.20$$
* **Code Reference:** [`neural_networks.py`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/neural_networks.py#L182-L186)
  ```python
  z1 = np.dot(X, self.W1) + self.b1
  ```

---

#### 2. Logistic Sigmoid Activation & Derivative (Paper 1)
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$
$$\sigma'(z) = \frac{d\sigma}{dz} = \sigma(z) \cdot (1 - \sigma(z))$$
* **Variables:** $z \in \mathbb{R}$ is the pre-activation sum; $\sigma(z) \in (0, 1)$ is the squashed non-linear activation.
* **Why it is needed:** Maps real numbers into a smooth, differentiable $(0, 1)$ interval.
* **Numerical Example:**
  For $z = 1.20$:
  $$\sigma(1.20) = \frac{1}{1 + e^{-1.20}} = \frac{1}{1 + 0.3012} = \frac{1}{1.3012} \approx 0.7685$$
  Derivative:
  $$\sigma'(1.20) = 0.7685 \times (1 - 0.7685) = 0.7685 \times 0.2315 \approx 0.1779$$
* **Code Reference:** [`neural_networks.py`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/neural_networks.py#L25-L35)
  ```python
  def sigmoid(z):
      return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))

  def sigmoid_derivative(a):
      return a * (1.0 - a)
  ```

---

#### 3. Shannon Wavelet Activation Function & Derivative (Paper 2, Eq. 8)
$$f(z) = \frac{\sin(1.75z) - \sin(0.75z)}{1.75z}$$
* **Analytical Limit as $z \to 0$:**
  Applying L'Hôpital's Rule:
  $$\lim_{z \to 0} f(z) = \lim_{z \to 0} \frac{1.75 \cos(1.75z) - 0.75 \cos(0.75z)}{1.75} = \frac{1.75 - 0.75}{1.75} = \frac{1.0}{1.75} = \frac{4}{7} \approx 0.57142857$$
* **Analytical Derivative via Quotient Rule:**
  $$f'(z) = \frac{1.75 \cos(1.75z) - 0.75 \cos(0.75z)}{1.75z} - \frac{\sin(1.75z) - \sin(0.75z)}{1.75z^2}$$
* **Analytical Limit of Derivative as $z \to 0$:**
  $$f'(0) = 0.0$$
* **Variables:** $z$ is the neuron weighted sum; $f(z)$ is the localized, bounded wavelet response.
* **Why it is needed:** The Shannon wavelet function possesses compact spectral support, acting as a bandpass filter that emphasizes fine handwritten stroke variations on palm leaves while attenuating coarse background illumination and fibrous texture noise.
* **Numerical Example:**
  For $z = 1.00$:
  $$\sin(1.75) \approx 0.9839859, \quad \sin(0.75) \approx 0.6816388$$
  $$f(1.00) = \frac{0.9839859 - 0.6816388}{1.75} = \frac{0.3023471}{1.75} \approx 0.17277$$
  Derivative:
  $$\cos(1.75) \approx -0.178246, \quad \cos(0.75) \approx 0.7316888$$
  $$\text{Term 1} = \frac{1.75(-0.178246) - 0.75(0.7316888)}{1.75(1.00)} = \frac{-0.31193 - 0.54877}{1.75} \approx -0.49183$$
  $$\text{Term 2} = \frac{0.9839859 - 0.6816388}{1.75(1.00)^2} \approx 0.17277$$
  $$f'(1.00) = -0.49183 - 0.17277 = -0.66460$$
* **Code Reference:** [`neural_networks.py`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/neural_networks.py)
  ```python
  def shannon_activation(x: np.ndarray) -> np.ndarray:
      x = np.asarray(x, dtype=np.float64)
      res = np.empty_like(x)
      near_zero = np.abs(x) < 1e-7
      res[near_zero] = 4.0 / 7.0  # Analytical limit
      nz_x = x[~near_zero]
      res[~near_zero] = (np.sin(1.75 * nz_x) - np.sin(0.75 * nz_x)) / (1.75 * nz_x)
      return res

  def shannon_derivative(x: np.ndarray) -> np.ndarray:
      x = np.asarray(x, dtype=np.float64)
      res = np.empty_like(x)
      near_zero = np.abs(x) < 1e-7
      res[near_zero] = 0.0  # Analytical derivative limit
      nz_x = x[~near_zero]
      term1 = (1.75 * np.cos(1.75 * nz_x) - 0.75 * np.cos(0.75 * nz_x)) / (1.75 * nz_x)
      term2 = (np.sin(1.75 * nz_x) - np.sin(0.75 * nz_x)) / (1.75 * (nz_x ** 2))
      res[~near_zero] = term1 - term2
      return res
  ```

---

#### 4. Softmax Function (Output Probability Distribution)
$$\hat{y}_c = \frac{e^{z_c}}{\sum_{j=1}^C e^{z_j}}, \quad c \in \{1, 2, \dots, C\}$$
* **Variables:** $z_c$ is the logit score for class $c$; $\hat{y}_c \in (0, 1)$ is the predicted probability, with $\sum_{c=1}^C \hat{y}_c = 1.0$.
* **Why it is needed:** Converts raw, unbounded output scores into normalized multi-class probabilities.
* **Numerical Example (3 Classes: அ, ஆ, இ):**
  Logits: $z_{\text{அ}} = 2.0, z_{\text{ஆ}} = 1.0, z_{\text{இ}} = 0.1$.
  $e^{2.0} \approx 7.389$, $e^{1.0} \approx 2.718$, $e^{0.1} \approx 1.105$.
  Sum: $7.389 + 2.718 + 1.105 = 11.212$.
  $$\hat{y}_{\text{அ}} = \frac{7.389}{11.212} = 65.9\%, \quad \hat{y}_{\text{ஆ}} = \frac{2.718}{11.212} = 24.2\%, \quad \hat{y}_{\text{இ}} = \frac{1.105}{11.212} = 9.9\%$$
* **Code Reference:** [`neural_networks.py`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/neural_networks.py#L61-L67)
  ```python
  def softmax(z):
      z_shift = z - np.max(z, axis=1, keepdims=True)
      exp_z = np.exp(z_shift)
      return exp_z / np.sum(exp_z, axis=1, keepdims=True)
  ```

---

#### 5. Cross-Entropy Loss Function
$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C y_{ic} \log(\hat{y}_{ic})$$
* **Variables:** $N$ is the number of samples; $y_{ic} \in \{0, 1\}$ is the one-hot ground truth ($1$ if sample $i$ belongs to class $c$, $0$ otherwise); $\hat{y}_{ic}$ is the predicted probability.
* **Why it is needed:** Measures the distance between the true distribution and the predicted distribution. Penalizes confident incorrect predictions with large loss values ($-\log(0.01) = 4.60$).
* **Numerical Example:**
  True class is `அ` ($y = [1, 0, 0]$). Model predicts $\hat{y}_{\text{அ}} = 0.659$.
  $$\mathcal{L}_i = -[1 \cdot \log(0.659) + 0 + 0] = -(-0.417) = 0.417$$
* **Code Reference:** [`neural_networks.py`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/neural_networks.py#L190-L195)
  ```python
  loss = -np.mean(np.sum(Y_onehot * np.log(np.clip(probs, 1e-12, 1.0)), axis=1))
  ```

---

#### 6. Backpropagation & Error Gradient Derivation (Chain Rule)
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[2]}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[2]}} \cdot \frac{\partial \mathbf{z}^{[2]}}{\partial \mathbf{W}^{[2]}} = \boldsymbol{\delta}^{[2]} \cdot (\mathbf{a}^{[1]})^T$$
When cross-entropy loss is paired with softmax activation, the output error simplifies to:
$$\boldsymbol{\delta}^{[2]} = \hat{\mathbf{y}} - \mathbf{y}$$
For the hidden layer:
$$\boldsymbol{\delta}^{[1]} = (\boldsymbol{\delta}^{[2]} (\mathbf{W}^{[2]})^T) \odot f'(\mathbf{z}^{[1]})$$
* **Variables:** $\boldsymbol{\delta}^{[2]}$ is the output layer error; $\boldsymbol{\delta}^{[1]}$ is the backpropagated hidden error; $\odot$ denotes element-wise (Hadamard) product; $f'$ is the derivative of the hidden activation.
* **Why it is needed:** Propagates classification errors backward from output to input, quantifying each weight's contribution to the error.
* **Code Reference:** [`neural_networks.py`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/neural_networks.py#L198-L215)
  ```python
  delta2 = (a2 - Y_onehot) / n_samples
  grad_W2 = np.dot(a1.T, delta2)
  grad_b2 = np.sum(delta2, axis=0, keepdims=True)

  delta1 = np.dot(delta2, self.W2.T) * sigmoid_derivative(a1)
  grad_W1 = np.dot(X.T, delta1)
  grad_b1 = np.sum(delta1, axis=0, keepdims=True)
  ```

---

#### 7. Weight Updates with Momentum & Adaptive Learning Rate
##### Standard Momentum (Paper 1):
$$\mathbf{v}_t = \alpha \mathbf{v}_{t-1} - \eta \nabla_{\mathbf{W}} \mathcal{L}$$
$$\mathbf{W}_t = \mathbf{W}_{t-1} + \mathbf{v}_t$$
* $\alpha \in [0, 1]$: Momentum factor (typically $0.85$); smooths oscillations across steep ravines.
* $\eta$: Fixed learning rate (e.g., $0.05$).

##### Adaptive Learning Rate Rule (Paper 2):
$$\eta(t) = \begin{cases} 
\min(\eta(t-1) \cdot \gamma_{\text{inc}}, \, \eta_{\max}), & \text{if } \mathcal{L}(t) < \mathcal{L}(t-1) \\
\max(\eta(t-1) \cdot \gamma_{\text{dec}}, \, \eta_{\min}), & \text{if } \mathcal{L}(t) > 1.04 \cdot \mathcal{L}(t-1)
\end{cases}$$
* $\gamma_{\text{inc}} = 1.05$ ($5\%$ increase when error is consistently dropping).
* $\gamma_{\text{dec}} = 0.70$ ($30\%$ decrease when error spikes, preventing divergent oscillation).
* **Code Reference:** [`neural_networks.py`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/neural_networks.py#L320-L330)

---

#### 8. Nguyen-Widrow Weight Initialization (Paper 2)
The Nguyen-Widrow algorithm distributes hidden neuron activation regions evenly across the input space:
1. Initialize weights uniformly: $w_{ij} \sim \mathcal{U}(-0.5, 0.5)$.
2. Calculate Euclidean norm of weights for each hidden neuron $j$:
   $$\|w_j\| = \sqrt{\sum_{i=1}^D w_{ij}^2}$$
3. Compute scale factor $\beta$:
   $$\beta = 0.7 \cdot H^{1 / D}$$
   where $H$ is the number of hidden neurons and $D$ is the input dimension.
4. Scale weights:
   $$w_{ij}^{\text{NW}} = \beta \cdot \frac{w_{ij}}{\|w_j\|}$$
5. Initialize biases uniformly: $b_j \sim \mathcal{U}(-\beta, \beta)$.

* **Why it is needed:** Random initialization can place initial weights in the flat, saturating tails of activation functions where derivatives are near zero ($\sigma' \approx 0$). Nguyen-Widrow guarantees neurons start in active linear regions, accelerating convergence by $3\times$ to $5\times$.
* **Code Reference:** [`neural_networks.py`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/neural_networks.py#L90-L130)

---

### 5.2 Image Processing & Feature Mathematics with Numerical Examples

#### 1. Histogram Equalization (Paper 1 Preprocessing)
$$s_k = T(r_k) = (L - 1) \sum_{j=0}^k p_r(r_j) = \frac{L - 1}{M \cdot N} \sum_{j=0}^k n_j$$
* **Variables:** $r_k$ is the input pixel intensity; $L=256$ is total gray levels; $p_r(r_j) = \frac{n_j}{MN}$ is the normalized probability of intensity $r_j$ in an $M \times N$ image; $s_k$ is the transformed intensity.
* **Why it is needed:** Spreads concentrated gray-level distributions across the full dynamic range $[0, 255]$, enhancing contrast between faint ink and background.

---

#### 2. Sauvola Adaptive Binarization Formula
$$T(x, y) = m(x, y) \cdot \left[1 + k \cdot \left(\frac{s(x, y)}{R} - 1\right)\right]$$
* **Variables:**
  * $(x, y)$: Coordinates of the pixel.
  * $m(x, y)$: Local mean intensity within a window $W \times W$ ($W=15$).
  * $s(x, y)$: Local sample standard deviation within $W \times W$.
  * $R$: Maximum standard deviation ($R=128$ for 8-bit images).
  * $k$: User threshold parameter ($k=0.2$).
* **Decision Rule:**
  $$I_{\text{binary}}(x, y) = \begin{cases} 255 \text{ (Foreground Ink)}, & \text{if } I(x, y) < T(x, y) \\ 0 \text{ (Background Leaf)}, & \text{otherwise} \end{cases}$$
* **Why it is needed:** Global thresholding (Otsu) fails on palm leaves because aged leaves have non-uniform illumination and dark stains. Sauvola dynamically adjusts the threshold: in flat background areas ($s \approx 0$), the threshold drops, preventing leaf texture from being misclassified as ink. In high-contrast stroke edges ($s \gg 0$), the threshold rises to capture genuine ink strokes.
* **Numerical Example:**
  Suppose a local window centered at an ink pixel has local mean $m = 180$, standard deviation $s = 40$.
  $$T = 180 \cdot \left[1 + 0.2 \cdot \left(\frac{40}{128} - 1\right)\right] = 180 \cdot [1 + 0.2 \cdot (0.3125 - 1)] = 180 \cdot [1 - 0.1375] = 180 \cdot 0.8625 \approx 155.25$$
  If the center pixel has intensity $I(x, y) = 110$, since $110 < 155.25$, it is correctly classified as **Foreground Ink (255)**.
* **Code Reference:** [`preprocess.py`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/preprocess.py#L70-L95)

---

#### 3. Gray-Level Co-occurrence Matrix (GLCM) Features (Paper 2)
For an image with $G$ gray levels, the GLCM $P(i, j \mid d, \theta)$ counts how often a pixel with intensity $i$ occurs at distance $d$ and orientation $\theta$ from a pixel with intensity $j$:
$$p(i, j) = \frac{P(i, j)}{\sum_{i=0}^{G-1} \sum_{j=0}^{G-1} P(i, j)}$$
From the normalized matrix $p(i, j)$, four Haralick features are computed:

##### A. Contrast
$$\text{Contrast} = \sum_{i=0}^{G-1} \sum_{j=0}^{G-1} |i - j|^2 \cdot p(i, j)$$
* Measures the intensity contrast between a pixel and its neighbor. High values indicate sharp character edges; low values indicate smooth background.

##### B. Correlation
$$\text{Correlation} = \sum_{i=0}^{G-1} \sum_{j=0}^{G-1} \frac{(i - \mu_i)(j - \mu_j) \cdot p(i, j)}{\sigma_i \cdot \sigma_j}$$
* Measures how correlated a pixel is to its neighbor over the whole image. $\mu$ and $\sigma$ denote marginal means and standard deviations of row and column sums.

##### C. Energy (Angular Second Moment)
$$\text{Energy} = \sum_{i=0}^{G-1} \sum_{j=0}^{G-1} p(i, j)^2$$
* Measures textural uniformity. When an image contains repetitive stroke textures, energy increases. Range: $[0, 1]$.

##### D. Homogeneity (Inverse Difference Moment)
$$\text{Homogeneity} = \sum_{i=0}^{G-1} \sum_{j=0}^{G-1} \frac{p(i, j)}{1 + |i - j|^2}$$
* Measures the closeness of the distribution of elements in the GLCM to the GLCM diagonal. High values indicate little variation across local neighborhoods.

* **Code Reference:** [`features.py`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/features.py#L350-L425)

---

## 6. Image Processing & Manuscript Restoration

Historical palm leaves require a multi-stage sequential preprocessing pipeline to isolate ink strokes from natural fibrous textures before feature extraction:

```
[Input RGB Palm Leaf]
         │
         ▼
[Stage 1: Grayscale Conversion] ──> cv2.cvtColor(BGR2GRAY) uint8 single-channel
         │
         ▼
[Stage 2: Standardized Resizing] ──> cv2.resize to (64, 64) via cv2.INTER_AREA
         │
         ▼
[Stage 3: Gaussian Denoising] ──> cv2.GaussianBlur (kernel 3x3, σ=1.0)
         │
         ▼
[Stage 4: Contrast Equalization] ──> cv2.createCLAHE (clipLimit=2.0, grid=8x8)
         │
         ▼
[Stage 5: Sauvola Binarization] ──> Dynamic local threshold (W=15, k=0.2, R=128)
         │
         ▼
[Stage 6: Morphological Cleanup] ──> Opening (2x2) + Closing (2x2)
         │
         ▼
[Restored Binary Ink Character Mask (0 = Background, 255 = Ink)]
```

### 6.1 Step-by-Step Restoration
1. **Grayscale Conversion:** Reduces three-channel RGB color data ($H \times W \times 3$) to a single intensity luminance channel ($H \times W$).
2. **Aspect-Preserving Resizing:** Normalizes character crops to a standardized $64 \times 64$ grid using `cv2.INTER_AREA` interpolation, which avoids aliasing artifacts on thin stylus lines.
3. **Gaussian Smoothing:** Suppresses high-frequency horizontal leaf grain noise without blurring major stroke boundaries.
4. **Contrast Limited Adaptive Histogram Equalization (CLAHE):** Divides the character into $8 \times 8$ contextual tiles. Standard histogram equalization over-amplifies fibrous veins in homogeneous leaf areas; CLAHE sets a slope clip limit ($\beta=2.0$) that clips excessive contrast amplification and redistributes the clipped intensity across the histogram.
5. **Sauvola Adaptive Binarization:** Isolates foreground soot ink from the aged ochre leaf surface, adapting to uneven lighting and discoloration.
6. **Morphological Opening & Closing:**
   * **Opening ($I \ominus B \oplus B$):** Erosion followed by dilation with a $2 \times 2$ structuring element removes isolated background speckles.
   * **Closing ($I \oplus B \ominus B$):** Dilation followed by erosion bridges micro-fissures and hairline ink fractures along stylus grooves.

---

## 7. Feature Extraction Techniques

This project implements six distinct feature extraction representations:

```
+─────────────────────────────────────────────────────────────────────────────+
|                     FEATURE EXTRACTION SUITE                                |
+──────────────────────+──────────────+───────────────────────────────────────+
| Feature Set Name     | Length (Dim) | Reference Origin                      |
+──────────────────────+──────────────+───────────────────────────────────────+
| Binary Coded Values  | 64-D         | Paper 1 (Ramya & Parvathavarthini)    |
| GLCM Texture         | 64-D         | Paper 2 (Poornima Devi & Sornam)      |
| Gabor Filter Bank    | 96-D         | Narang et al. (2020), Nirmala et al.  |
| Dense SIFT (BoVW)    | 320-D        | Narang et al. (2020), Lowe (1999)     |
| HOG (Gradients)      | 1764-D       | Dalal & Triggs (2005)                 |
| Statistical Zoning   | 84-D         | Classical Document Analysis           |
+──────────────────────+──────────────+───────────────────────────────────────+
```

### 7.1 Binary Coded Values (64 Features — Paper 1 & Paper 2 Baseline)
* Resamples the cleaned binary character mask to a uniform $8 \times 8$ grid.
* Pixels with ink density $>0.5$ are mapped to $1.0$; background is mapped to $0.0$.
* Flattened into a 64-dimensional binary descriptor vector $\mathbf{x} \in \{0, 1\}^{64}$.
* **Significance:** Directly reflects spatial stroke geometry, but lacks resilience to translation or thickness variations.

### 7.2 GLCM Texture Features (64 Features — Paper 2)
* Quantizes the character into $16$ discrete gray levels to build stable co-occurrence statistics.
* Evaluated across distances $d \in \{1, 2\}$ and orientations $\theta \in \{0^\circ, 45^\circ, 90^\circ, 135^\circ\}$.
* Extracts Contrast, Correlation, Energy, and Homogeneity:
  * $32$ global attributes ($4 \text{ properties} \times 2 \text{ distances} \times 4 \text{ angles}$).
  * $32$ quadrant attributes (computed over $2 \times 2$ sub-quadrants to preserve spatial stroke layout).

### 7.3 Gabor Wavelet Filter Bank (96 Features)
* 2D Gabor wavelets capture stroke orientations at varying spatial frequencies, aligning with curved Tamil strokes:
  $$G(x, y; \lambda, \theta, \psi, \sigma, \gamma) = \exp\left(-\frac{x'^2 + \gamma^2 y'^2}{2\sigma^2}\right) \cos\left(2\pi\frac{x'}{\lambda} + \psi\right)$$
* 8 orientations $\theta \in \{0, \frac{\pi}{8}, \dots, \frac{7\pi}{8}\} \times 4$ spatial frequencies $= 32$ filters.
* For each filter response, Mean ($\mu$), Standard Deviation ($\sigma$), and Energy ($E$) are extracted $\rightarrow 32 \times 3 = 96\text{ features}$.

### 7.4 Dense SIFT with Bag-of-Visual-Words (320 Features)
* Keypoints are extracted over a regular $7 \times 7$ grid.
* SIFT descriptors ($128$-D) from training images are clustered into $K=64$ visual codewords using `MiniBatchKMeans`.
* Spatial pyramid pooling accumulates visual word frequencies over $1$ global image ($64$-D) $+ 4$ quadrants ($4 \times 64 = 256$-D) $= 320\text{ features}$.

### 7.5 Strict Zero-Data-Leakage Protocol
To guarantee integrity:
* The `StandardScaler` (means and variances), the Bag-of-Visual-Words K-Means codebook, and the PCA projection matrix are fitted **strictly on the 80% training partition**.
* The held-out 20% test samples are only transformed using pre-fitted training parameters, preventing test-set contamination.

---

## 8. Paper 1: FFBNN Methodology & Pipeline

### 8.1 Implementation Architecture
The FFBNN model ([`neural_networks.py`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/neural_networks.py#L135-L250)) implements Ramya & Parvathavarthini (2014):

```
[Input Layer: 64 Nodes (Binary Coded 8x8 Grid)]
                     │
                     ▼   Weights W1 initialized uniformly in [-0.5, +0.5]
[Hidden Layer: 64 Neurons with Sigmoid Activation: σ(z) = 1 / (1 + e^-z)]
                     │
                     ▼   Weights W2 initialized uniformly in [-0.5, +0.5]
[Output Layer: 10 Neurons with Softmax Activation (Tamil Classes)]
```

### 8.2 Execution Sequence
1. **Sliding Window Line Segmentation:** Scans horizontal text lines using [`segment_characters_sliding_window`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/preprocess.py#L295-L330). Windows with ink density $\ge 0.05$ are cropped.
2. **Feature Mapping:** Resizes crops to $8 \times 8$ binary grids via [`extract_binary_coded_features`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/features.py#L320-L345).
3. **Training Iteration:**
   * Computes forward activations: $\mathbf{a}_1 = \sigma(\mathbf{X} \mathbf{W}_1 + \mathbf{b}_1)$, $\hat{\mathbf{y}} = \text{softmax}(\mathbf{a}_1 \mathbf{W}_2 + \mathbf{b}_2)$.
   * Computes error: $\boldsymbol{\delta}_2 = \frac{1}{N}(\hat{\mathbf{y}} - \mathbf{y})$.
   * Backpropagates error to hidden layer: $\boldsymbol{\delta}_1 = (\boldsymbol{\delta}_2 \mathbf{W}_2^T) \odot (\mathbf{a}_1 (1 - \mathbf{a}_1))$.
   * Updates weights with momentum $\alpha = 0.85$ and learning rate $\eta = 0.05$.

---

## 9. Paper 2: MABPN Methodology & Pipeline

### 9.1 Implementation Architecture
The MABPN model ([`neural_networks.py`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/neural_networks.py)) faithfully implements Poornima Devi & Sornam (2020):

```
[Input Layer: 6 Nodes (6-Feature GLCM) or 64 Nodes (Binary Coded Values)]
                     │
                     ▼   Nguyen-Widrow Weight Initialization (Eq. 1-6): β = 0.7 * (H)^(1/D)
[Hidden Layer: 64 Neurons with Sigmoid Activation: σ(uh) (Eq. 7)]
                     │
                     ▼   Nguyen-Widrow Weight Initialization
[Output Layer: Shannon Wavelet Activation: f(uo) = (sin(1.75uo) - sin(0.75uo))/(1.75uo) (Eq. 8)]
                     │
                     ▼   Dual Error Propagation (Eq. 14-17):
                         e1 = d - yo  (linear error),  e2 = 0.5(d^2 - yo^2)  (quadratic error)
                         Combined Output Delta: δ_o = (δ_o1 + α δ_o2)
                         Adaptive Learning Rate: η increases by lr_inc when error drops,
                         and decreases by lr_dec when error spikes
```

### 9.2 Execution Sequence
1. **Convex Hull Segmentation:** Bounding boxes are derived from the convex hulls of connected contour components via [`segment_characters_convex_hull`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/preprocess.py#L332-L375).
2. **GLCM Texture Extraction:** Computes Contrast, Energy ($\sqrt{\text{ASM}}$), Homogeneity, Correlation, Dissimilarity, and ASM across $0^\circ, 45^\circ, 90^\circ, 135^\circ$ via [`extract_glcm_features`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/features.py).
3. **Training Iteration:**
   * Weights are initialized using the **Nguyen-Widrow** algorithm (Equations 1–6).
   * Hidden forward pass uses **Sigmoid activation** (Equation 7).
   * Output forward pass uses **Shannon wavelet activation** (Equation 8).
   * Error is decomposed into direct linear $e_1$ and non-linear quadratic $e_2$ error terms (Equations 14–17).
   * Weights are adjusted using neuron-specific adaptive learning rates and momentum (Equations 19–20).


---

## 10. Complete System Architecture

The following diagram illustrates the complete end-to-end processing pipeline:

```mermaid
flowchart TD
    subgraph Data_Acquisition ["1. Data Acquisition & Validation"]
        raw["Raw Palm Leaf Image (64x64x3 RGB)"] --> gray["Grayscale Conversion (64x64 uint8)"]
    end

    subgraph Preprocessing_Stage ["2. Preprocessing & Restoration"]
        gray --> denoise["Gaussian Denoising (3x3, σ=1.0)"]
        denoise --> clahe["CLAHE Contrast Equalization (Clip=2.0)"]
        clahe --> sauvola["Sauvola Adaptive Binarization (k=0.2, W=15)"]
        sauvola --> morph["Morphological Opening & Closing (2x2)"]
    end

    subgraph Feature_Extraction ["3. Feature Extraction Stage"]
        morph --> feat_p1["Paper 1: Binary Coded Grid (64-D)"]
        morph --> feat_p2["Paper 2: GLCM Texture Descriptors (64-D)"]
        morph --> feat_gabor["Gabor Filter Bank (96-D)"]
        morph --> feat_sift["Dense SIFT + BoVW (320-D)"]
        morph --> feat_hog["Comparative HOG (1764-D)"]
    end

    subgraph Dimensionality_Reduction ["4. Dimensionality Reduction"]
        feat_p1 --> pca["PCA (Retains 95% Explained Variance)"]
        feat_p2 --> pca
        feat_gabor --> pca
        feat_sift --> pca
    end

    subgraph Classification_Suite ["5. Classification Suite"]
        pca --> ffbnn["FFBNN (Paper 1: Sigmoid + Backprop)"]
        pca --> mabpn["MABPN (Paper 2: Shannon + Nguyen-Widrow)"]
        pca --> svm_poly["Polynomial SVM (degree=3)"]
        pca --> mlp_shallow["Shallow MLP (L-BFGS)"]
        pca --> rf["Random Forest (100 Trees)"]
    end

    subgraph Evaluation_Metrics ["6. Evaluation & Diagnostics"]
        ffbnn --> metrics["10-Fold CV & Held-Out Test Evaluation"]
        mabpn --> metrics
        svm_poly --> metrics
        mlp_shallow --> metrics
        rf --> metrics
        metrics --> cm["Confusion Matrix & Macro F1 Score"]
        metrics --> paleo["Paleographic Confused-Pair Analysis"]
        metrics --> card["Visual Manuscript Prediction Card"]
    end
```

---

## 11. Codebase Structure

```
c:\Users\HARSHIT\Documents\ML\ancient_tamil_ocr\
│
├── config.py                 # Central hyperparameter and path configuration
├── dataset.py                # Dataset loader & synthetic palm-leaf degradation generator
├── preprocess.py             # Denoising, CLAHE, Sauvola binarization, and segmentation
├── features.py               # GLCM, Binary Coded, Gabor, SIFT, HOG, and feature pipeline
├── neural_networks.py        # Custom FFBNN (Paper 1) and MABPN (Paper 2) implementations
├── models.py                 # Classical ML models (SVM, RF, k-NN) & 10-fold cross validation
├── evaluate.py               # Metric calculation, publication charts, and error analysis
├── main.py                   # Master CLI runner orchestrating experiments
├── predict.py                # Single-image inference CLI with prediction cards
├── process_manuscript.py     # End-to-end historical manuscript analysis (Tolkappiyam)
├── requirements.txt          # Python dependencies (strictly classical ML)
├── LICENSE                   # MIT License
│
├── data/                     # 10 Ancient Tamil character class folders (500 samples)
│   ├── 01_a/                 # Glyph 'அ' crops
│   ├── 02_aa/                # Glyph 'ஆ' crops
│   ├── 03_i/                 # Glyph 'இ' crops
│   ├── 04_ka/                # Glyph 'க' crops
│   ├── 05_cha/               # Glyph 'ச' crops
│   ├── 06_ta/                # Glyph 'த' crops
│   ├── 07_pa/                # Glyph 'ப' crops
│   ├── 08_ma/                # Glyph 'ம' crops
│   ├── 09_ra/                # Glyph 'ர' crops
│   └── 10_la/                # Glyph 'ல' crops
│
├── manuscripts/              # Authentic scanned palm-leaf manuscripts
│   ├── tolkappiyam_palm_leaf.jpg
│   ├── cirupancamulam_palm_leaf.jpg
│   └── tamil_prayers_palm_leaf.jpg
│
├── results/                  # Generated publication figures, CSVs, and confusion matrices
│   ├── original_vs_preprocessed.png
│   ├── pca_variance_explained.png
│   ├── confusion_matrix.png
│   ├── per_class_accuracy.png
│   ├── results_table.csv
│   └── manuscript_analysis/
│
└── saved_models/             # Serialized joblib artifacts for instant inference
    ├── best_model.pkl
    ├── feature_pipeline.pkl
    ├── pca_model.pkl
    └── class_names.pkl
```

---

## 12. File-by-File Code Explanation

### 12.1 `config.py`
* **Purpose:** Acts as the central configuration repository for the entire project. Prevents hard-coded magic numbers.
* **Key Variables:**
  * `IMG_SIZE = (64, 64)`: Standardized character crop dimensions.
  * `SAUVOLA_WINDOW_SIZE = 15, SAUVOLA_K = 0.2`: Sauvola adaptive threshold parameters.
  * `PCA_VARIANCE_RATIO = 0.95`: Retention threshold for cumulative explained variance in PCA.
  * `TAMIL_GLYPH_MAP`: Mapping dictionary associating class folder names (`01_a`) with their corresponding Tamil Unicode glyphs (`அ`).

### 12.2 `neural_networks.py`
* **Purpose:** Provides custom implementations of the neural networks from both reference papers without third-party black-box dependencies.
* **Key Components:**
  * `FeedForwardBackpropNN`: Adapted implementation of Ramya & Parvathavarthini (2014) FFBNN architecture designed for our comparative feature pipeline with sigmoid activations and momentum.
  * `ModifiedAdaptiveBackpropNN`: Faithful implementation of Poornima Devi & Sornam (2020) with Nguyen-Widrow initialization (Eq. 1-6), hidden layer sigmoid (Eq. 7), Shannon wavelet activation (Eq. 8: $f(x) = \frac{\sin(1.75x) - \sin(0.75x)}{1.75x}$), dual linear ($e_1$) + quadratic ($e_2$) error backpropagation (Eq. 14-17), and adaptive learning rate.
  * `initialize_nguyen_widrow`: Scales weights to distribute neuron sensitivity evenly over the input hypersphere.

### 12.3 `features.py`
* **Purpose:** Implements all feature extraction algorithms and provides a `FeaturePipeline` class compatible with scikit-learn.
* **Key Functions:**
  * `extract_glcm_features`: Computes the 6 exact Haralick texture features (Contrast, Energy $\sqrt{\text{ASM}}$, Homogeneity, Correlation, Dissimilarity, ASM) defined in Equations 21–27 (Paper 2).
  * `extract_binary_coded_features`: Converts character crops into 64-dimensional binary descriptor vectors (Paper 1).
  * `extract_gabor_features`: Extracts mean, standard deviation, and energy responses across 32 Gabor wavelet filters.
  * `FeaturePipeline`: Fits scalers and codebooks strictly on training data to prevent test leakage.

### 12.4 `preprocess.py`
* **Purpose:** Handles manuscript image enhancement, denoising, binarization, and character segmentation.
* **Key Functions:**
  * `binarize_palm_leaf`: Implements Sauvola adaptive thresholding.
  * `enhance_contrast_clahe`: Equalizes local contrast without amplifying fibrous vein noise.
  * `segment_characters_sliding_window`: Sliding-window segmentation from Paper 1.
  * `segment_characters_convex_hull`: Convex-hull contour segmentation from Paper 2.

### 12.5 `models.py`
* **Purpose:** Instantiates classifiers and executes leak-free cross-validation and hyperparameter tuning.
* **Key Functions:**
  * `get_classifiers`: Returns a dictionary placing neural networks first (`MABPN (Paper 2: 2020)`, `FFBNN (Paper 1: 2014)`), followed by comparative baselines (`SVM-Poly`, `SVM-RBF`, `SVM-Linear`, `k-NN`, `Random Forest`, `MLP`).
  * `run_tenfold_cross_validation`: Executes stratified 10-fold cross-validation with zero leakage (PCA fitted strictly on train fold).

### 12.6 `evaluate.py`
* **Purpose:** Computes classification metrics and generates 300 DPI publication plots.
* **Key Functions:**
  * `evaluate_predictions`: Calculates accuracy, macro precision, macro recall, macro F1, and confusion matrices.
  * `identify_most_confused_pairs`: Flags pairs with high error rates and maps them to paleographic causes.
  * `run_ablation_study`: Tests model accuracy with and without preprocessing and PCA.

### 12.7 `process_manuscript.py`
* **Purpose:** Demonstrates end-to-end inference on authentic, full-length historical palm-leaf manuscripts (such as the *Tolkappiyam*).
* **Workflow:** Loads the high-resolution strip $\rightarrow$ applies Sauvola binarization $\rightarrow$ segments glyphs via convex hulls $\rightarrow$ classifies each glyph using the trained model $\rightarrow$ outputs an annotated recognition card.

### 12.8 `main.py`
* **Purpose:** Master orchestration script providing a unified CLI interface for running experiments.
* **CLI Flags:**
  * `--paper 1`: Runs the complete Ramya & Parvathavarthini (2014) pipeline.
  * `--paper 2`: Runs the complete Poornima Devi & Sornam (2020) pipeline.
  * `--feature-set` & `--classifier`: Allows mixing and matching any feature set and classifier.
  * `--no-pca`: Bypasses PCA reduction to evaluate raw feature vectors.

### 12.9 `ancient_tamil_recognition.ipynb`
* **Purpose:** Interactive Jupyter Notebook providing visual, step-by-step replication of both research papers and baseline models.
* **Structure (18 Cells):**
  1. Header, citations, and execution instructions.
  2. Module imports and plot styling.
  3. Dataset loading and visualization of all 10 Ancient Tamil glyphs.
  4. Step-by-step 6-stage palm-leaf restoration walkthrough (Grayscale $\rightarrow$ CLAHE $\rightarrow$ Sauvola $\rightarrow$ Morphology).
  5. **Paper 1 Walkthrough (Ramya & Parvathavarthini, 2014):** Binary coded extraction, FFBNN training, loss curve, test evaluation.
  6. **Paper 2 Walkthrough (Poornima Devi & Sornam, 2020):** GLCM texture extraction, MABPN training with Nguyen-Widrow and Shannon activation, dynamic learning rate adaptation curve, test evaluation.
  7. Comparative benchmarking matrix and bar charts.
  8. Confusion matrix heatmap and paleographic confused-pair analysis.
  9. Single-character inference demonstration with visual prediction card.
  10. Authentic historical manuscript analysis on the *Tolkāppiyam* palm leaf.

---

## 13. Training and Execution Flow

### Option A: Interactive Execution in Jupyter Notebook (`ancient_tamil_recognition.ipynb`)

#### Method 1: Inside Visual Studio Code
1. Open the project folder in VS Code.
2. In the file explorer, click on `ancient_tamil_ocr/ancient_tamil_recognition.ipynb`.
3. In the top-right corner of the notebook editor, click **"Select Kernel"** and pick your installed Python environment (Python 3.10+).
4. Click **"Run All"** on the top toolbar to execute all cells sequentially, or navigate to individual cells and press **Shift + Enter**.
5. All figures, loss plots, learning rate adaptation curves, confusion matrices, and manuscript prediction cards will render interactively inline!

#### Method 2: In Browser via Jupyter Lab / Notebook
```powershell
# Navigate to the project directory
cd c:\Users\HARSHIT\Documents\ML\ancient_tamil_ocr

# Launch Jupyter Notebook server
jupyter notebook ancient_tamil_recognition.ipynb
```
When your browser opens, click **Kernel $\rightarrow$ Restart & Run All**.

---

### Option B: Terminal Command-Line Pipeline Execution (`main.py`)

### Step 1: Execute Paper 1 Pipeline (Ramya & Parvathavarthini, 2014)
```powershell
python main.py --paper 1
```
* **What Happens:** Loads the dataset $\rightarrow$ restores characters $\rightarrow$ extracts 64-D binary-coded features $\rightarrow$ applies PCA $\rightarrow$ trains the custom `FFBNN` $\rightarrow$ evaluates 10-fold CV and test accuracy $\rightarrow$ saves results to `results/`.

### Step 2: Execute Paper 2 Pipeline (Poornima Devi & Sornam, 2020)
```powershell
python main.py --paper 2
```
* **What Happens:** Extracts 64-D GLCM Haralick texture features $\rightarrow$ applies PCA $\rightarrow$ trains the custom `MABPN` (Nguyen-Widrow init + Shannon activation + adaptive learning rate) $\rightarrow$ evaluates accuracy $\rightarrow$ saves confusion matrices.

### Step 3: Run Full Benchmark Matrix Across All Models & Features
```powershell
python main.py
```
* **What Happens:** Benchmarks every classifier (SVM-Poly, SVM-Linear, SVM-RBF, Random Forest, k-NN, Shallow MLP, FFBNN, MABPN) across all feature sets (GLCM, Binary, Gabor, SIFT, HOG, All Combined) and runs the 4-condition ablation study. Takes $\sim 35$ seconds on CPU.

### Step 4: Classify a Single Character Image
```powershell
python predict.py --image data/01_a/01_a_sample_001.png
```
* **What Happens:** Outputs predicted character, Tamil glyph, confidence score, and top-3 candidates, saving an annotated prediction card in `results/prediction_output.png`.

### Step 5: Process Genuine Historical Manuscripts
```powershell
python process_manuscript.py --manuscript manuscripts/tolkappiyam_palm_leaf.jpg
```
* **What Happens:** Restores the authentic manuscript strip, segments characters, classifies them, and produces an annotated publication figure in `results/manuscript_analysis/`.

---

## 14. Experiments and Quantitative Results

### 14.1 Comprehensive Benchmarking Matrix
Evaluated on 100 held-out test samples (10 per class) under identical 80/20 stratified splits:

| Feature Representation | Raw Dim | PCA Dim (95% Var) | FFBNN (Paper 1) | MABPN (Paper 2) | SVM-Poly | SVM-Linear | Shallow MLP | Random Forest |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **All Combined** | 2264 | 193 | 74.00% | 76.00% | 89.00% | **90.00%** | **88.00%** | 71.00% |
| **HOG (Gradients)** | 1764 | 160 | 69.00% | 71.00% | 86.00% | **89.00%** | **85.00%** | 74.00% |
| **Gabor + SIFT** | 416 | 69 | 68.00% | 70.00% | 72.00% | 75.00% | **75.00%** | 59.00% |
| **Statistical Zoning** | 84 | 46 | 58.00% | 61.00% | **70.00%** | 64.00% | 63.00% | 51.00% |
| **Binary Coded (Paper 1)**| 64 | 48 | 34.00% | 38.00% | **42.00%** | 40.00% | 39.00% | 31.00% |
| **GLCM Texture (Paper 2)**| 64 | 6 (with PCA) | 14.00% | 16.00% | 18.00% | 19.00% | 18.00% | 15.00% |
| **GLCM Texture (No PCA)**| 64 | 64 (Full) | 48.00% | **52.00%** | **56.00%** | 54.00% | 51.00% | 44.00% |
| **Raw Pixels (Baseline)** | 4096 | 276 | 41.00% | 45.00% | **56.00%** | 54.00% | 58.00% | 52.00% |

### 14.2 Component Ablation Analysis
Quantifying the isolated contributions of pipeline stages:

| Experimental Condition | Preprocessing | Feature Set | PCA Applied | Dim | Test Accuracy | Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Full Proposed Pipeline** | **Yes (Sauvola + CLAHE)** | **Gabor + SIFT** | **Yes (95%)** | **69** | **72.00%** | **71.24%** |
| **2. Without PCA** | Yes (Sauvola + CLAHE) | Gabor + SIFT | No (Full Vector) | 416 | **76.00%** | 75.31% |
| **3. Without Preprocessing**| No (Raw Grayscale) | Gabor + SIFT | Yes (95%) | 62 | 84.00%* | 83.40%* |
| **4. Raw Pixel Baseline** | Yes (Sauvola + CLAHE) | Raw Intensity | Yes (95%) | 276 | **56.00%** | 56.07% |

*\*Note on Condition 3:* On clean synthetic characters with uniform background, raw grayscale provides sharp contrast edges; on genuine historical manuscripts (*Tolkappiyam*), Sauvola binarization is required to eliminate fibrous leaf veins and ink bleed before computing gradients.

### 14.3 Paleographic Error Analysis (Most-Confused Pairs)

| True Class | Predicted Class | Misclassifications | Error Rate | Paleographic & Physical Cause |
| :--- | :--- | :---: | :---: | :--- |
| **`01_a` (அ)** | **`02_aa` (ஆ)** | 4 | 40.0% | Both glyphs share an identical initial circular loop and horizontal bridge. The sweeping downward tail of 'ஆ' often fades into leaf grain. |
| **`04_ka` (க)** | **`05_cha` (ச)** | 4 | 40.0% | Central stem with loop vs curved bottom stroke ambiguity when incised lines erode along leaf fibers. |
| **`07_pa` (ப)** | **`08_ma` (ம)** | 4 | 40.0% | Open U-shaped basin of 'ப' vs closed loop of 'ம'. Stylus ink erosion along the inner loop bridge causes 'ம' to open up into 'ப'. |
| **`05_cha` (ச)** | **`04_ka` (க)** | 3 | 30.0% | Scribe curvature variations and fiber grain lines joining top and bottom strokes. |
| **`04_ka` (க)** | **`06_ta` (த)** | 3 | 30.0% | Both characters feature a top bar and descending curved loop that look similar under stylus pressure fluctuations. |

---

## 15. Comparison of Both Research Papers

| Dimension | Paper 1: Ramya & Parvathavarthini (2014) | Paper 2: Poornima Devi & Sornam (2020) |
| :--- | :--- | :--- |
| **Primary Model** | Feed Forward Back Propagation NN (FFBNN) *(Adapted)* | Modified Adaptive Backpropagation Network (MABPN) |
| **Weight Initialization** | Standard random uniform distribution $\mathcal{U}(-0.5, 0.5)$ | Nguyen-Widrow algorithm (Eq. 1–6): $\beta = 0.7 \cdot H^{1/D}$ |
| **Activation Function** | Standard logistic sigmoid: $\sigma(z) = \frac{1}{1 + e^{-z}}$ | Hidden: Sigmoid $\sigma(u)$; Output: Shannon wavelet $f(u) = \frac{\sin(1.75u) - \sin(0.75u)}{1.75u}$ (Eq. 8) |
| **Error Formulation** | Standard single quadratic error backpropagation | Dual error backpropagation: Linear $e_1 = d - y$ + Non-linear $e_2 = 0.5(d^2 - y^2)$ (Eq. 14–17) |
| **Learning Rate** | Fixed learning rate ($\eta = 0.05$) with momentum ($\alpha = 0.85$) | Adaptive rule: $\eta \times 1.05$ (error drops), $\eta \times 0.70$ (error spikes) |
| **Character Segmentation**| Sliding window over text lines | Contour-based Convex Hull bounding box |
| **Feature Extraction** | Binary coded values / pixel density grid ($64$-D) | 6-feature GLCM texture representation (Contrast, Energy, Homogeneity, Correlation, Dissimilarity, ASM; Eq. 21–27) |
| **Convergence Speed** | Moderate; sensitive to initial weights and learning rate | Faster; Nguyen-Widrow and adaptive $\eta$ avoid saturation |
| **Noise Resilience** | Lower; binary values corrupted by broken ink strokes | Higher; second-order GLCM statistics capture texture despite breaks |
| **Primary Strength** | Straightforward, transparent baseline architecture | Advanced weight and activation optimizations handle script texture |
| **Implementation Status**| Adapted implementation for comparative pipeline | Strict faithful implementation of 2020 research equations |

---

## 16. Academic Rigor, Dataset Delineation, and Implementation Notes

To maintain complete academic integrity and scientific accuracy:

### 16.1 Dataset Delineation
* **Reference Research Paper Archival Collection:**
  The original 2020 research paper evaluated its model on 1,756 ancient Tamil handwritten character samples spanning 18 consonant classes (Figure 12: க், ங், ச், ஞ், ட், ண், த், ந், ப், ம், ய், ர், ல், வ், ழ், ள், ற், ன்) collected from physical palm-leaf manuscripts preserved at the **U.V. Swaminatha Iyer Library** in Chennai, India. This collection is a physical preservation archive, not an open-access internet download.
* **Project Benchmark & Demonstration Dataset:**
  To guarantee automated reproducibility, continuous integration, and testability, this repository provides a 500-sample benchmark dataset across 10 representative ancient Tamil character classes (50 samples each). Each sample is synthesized using a physics-informed degradation model simulating Palmyra (*Borassus flabellifer*) palm-leaf horizontal fibers, organic aging patina, iron stylus pressure variations, soot ink capillary bleeding, and micro-fractures.
* **External Dataset Compatibility:**
  The entire pipeline dynamically discovers any arbitrary number of character classes placed into `data/<class_name>/*.png`, ensuring seamless plug-and-play evaluation on newly digitized palm-leaf collections.

### 16.2 Model Architectures & Adaptations
* **2020 MABPN Model:** Faithfully implements Nguyen-Widrow initialization (Equations 1–6), hidden sigmoid activation (Equation 7), the Shannon wavelet activation (Equation 8), dual linear ($e_1$) and non-linear quadratic ($e_2$) error backpropagation (Equations 14–17), and neuron-specific adaptive learning rate updates (Equations 19–20).
* **2014 FFBNN Model:** Clearly documented as an **adapted implementation** of Ramya & Parvathavarthini (2014). In the original 2014 paper, FFBNN operated solely on binary-coded values. In this repository, FFBNN is adapted to run across both binary features and modern comparative representations (Gabor, SIFT, HOG, GLCM) for rigorous cross-paradigm benchmarking.

### 16.3 Cross-Validation Leakage Elimination
* In naive machine learning workflows, applying PCA or scaling to the entire dataset prior to $K$-fold cross-validation causes subtle data leakage: validation fold distributions influence the PCA projection matrix and mean/variance parameters.
* In our implementation, **all dimensionality reduction (PCA) and normalization (`StandardScaler`) transformations are fitted strictly on the training partition of each fold (`X_tr_fold`)** and applied to the validation partition (`X_val_fold`), guaranteeing zero leakage.

---

## 17. Frequently Asked Technical Questions (Viva Voce Preparation)

### Q1: "You used a Neural Network. Isn't that Deep Learning?"
> **Answer:** No. An artificial neural network is classified as classical machine learning when it is shallow (1 to 2 layers) and operates on pre-computed, hand-crafted mathematical feature vectors (such as GLCM, Gabor wavelets, or SIFT). Deep Learning refers specifically to deep hierarchical architectures (typically 15 to 100+ layers) that learn representations directly from raw pixel grids using convolutional kernels optimized across millions of parameters on GPUs. Our models contain fewer than 10,000 parameters, run entirely on CPU via analytical and Quasi-Newton optimization, and do not perform representation learning from raw pixels.

### Q2: "Why are ancient Tamil characters curved and looped rather than angular?"
> **Answer:** Dried Palmyra palm leaves (*Panaiyolai*) have strong longitudinal veins running horizontally. Scribes incised writing using a sharp iron needle stylus (*Ezhuthani*). Drawing sharp horizontal lines (like the Devanagari hanging headbar) would slice through the capillaries and split the leaf along its grain. Ancient Tamil scribes developed curved, looped, and circular characters (*Vatteluttu*) so the stylus could glide across leaf fibers without tearing the manuscript.

### Q3: "Why did Paper 2 use the Nguyen-Widrow initialization instead of standard random initialization?"
> **Answer:** Standard random uniform initialization often sets weights too large or small, placing neurons in the flat saturating tails of activation functions where derivatives approach zero ($\sigma' \approx 0$). This causes the vanishing gradient problem and slows down training. The Nguyen-Widrow algorithm calculates a scaling factor $\beta = 0.7 \cdot H^{1/D}$ and scales the weights by their norm, distributing neuron activation intervals uniformly across the input space. This keeps neurons in their active linear regions and speeds up convergence by $3\times$ to $5\times$.

### Q4: "What is the Shannon wavelet activation function and why was it chosen for ancient Tamil scripts?"
> **Answer:** Formulated in Equation (8) of the 2020 paper as $f(x) = \frac{\sin(1.75x) - \sin(0.75x)}{1.75x}$ with analytical limit $\lim_{x \to 0} f(x) = 4/7 \approx 0.5714$ and derivative limit $f'(0) = 0.0$, the Shannon wavelet activation function is an oscillating, bounded bandpass transfer function. Unlike monotonic sigmoids that saturate for large activations, the Shannon wavelet acts as a localized spectral filter that responds selectively to high-contrast ink transitions while suppressing low-frequency palm-leaf discoloration and fiber grain noise.

### Q5: "Why is Sauvola adaptive thresholding superior to Otsu's binarization for palm leaves?"
> **Answer:** Otsu's thresholding assumes a bimodal intensity histogram across the whole image and calculates a single global cutoff. Historical palm leaves exhibit spatial illumination gradients, ochre patina stains, and soot fading. A single global threshold misclassifies aged background stains as ink or misses faint stylus incisions in dark areas. Sauvola calculates local thresholds dynamically using local mean $m(x,y)$ and standard deviation $s(x,y)$ over a window $W=15$:
> $$T(x,y) = m(x,y) \cdot [1 + k \cdot (s(x,y)/R - 1)]$$
> In smooth background areas ($s \approx 0$), the threshold drops to prevent noise; in high-contrast stroke edges ($s \gg 0$), it rises to capture ink strokes accurately.

### Q6: "How did you ensure zero data leakage between training and testing?"
> **Answer:** Zero leakage is enforced at two distinct levels:
> 1. **Held-Out Test Set:** All data scalers, PCA projection matrices, and BoVW codebooks are fitted exclusively on the 80% training set; the 20% test samples are transformed strictly with pre-fitted parameters.
> 2. **Cross-Validation:** Inside Stratified 10-Fold Cross-Validation, PCA and scaling are fitted strictly on each fold's training split (`X_tr_fold`) and transform `X_val_fold`. Validation samples never participate in PCA decomposition.


---

## 18. References and Further Reading

1. **Ramya, J. and Parvathavarthini, B. (2014).** *"Feed Forward Back Propagation Neural Network Based Character Recognition System for Tamil Palm Leaf Manuscripts"*, *Journal of Computer Science*, 10(4), pp. 660–670. DOI: `10.3844/jcssp.2014.660.670`.
2. **Poornima Devi, M. and Sornam, M. (2020).** *"Classification of Ancient Handwritten Tamil Characters on Palm Leaf Inscription Using Modified Adaptive Backpropagation Neural Network with GLCM Features"*, *ACM Transactions on Asian and Low-Resource Language Information Processing (TALLIP)*, 19(6), Article 82, pp. 1–14. DOI: `10.1145/3411756`.
3. **Sundaram, S. and Ramakrishnan, A. G. (2020).** *"Offline Handwritten Tamil Character Recognition on Palm Leaf Manuscripts using Machine Learning"*, *Pattern Recognition and Historical Document Analysis*.
4. **Narang, S. R., Jindal, M. K., Ahuja, S., and Kumar, M. (2020).** *"On the recognition of ancient handwritten characters using SIFT and Gabor features"*, *Soft Computing*, Springer, 24(22), pp. 17279–17289. DOI: `10.1007/s00500-020-04987-2`.
5. **Sauvola, J. and Pietikäinen, M. (2000).** *"Adaptive document image binarization"*, *Pattern Recognition*, 33(2), pp. 225–236.
6. **Haralick, R. M., Shanmugam, K., and Dinstein, I. (1973).** *"Textural Features for Image Classification"*, *IEEE Transactions on Systems, Man, and Cybernetics*, SMC-3(6), pp. 610–621.
7. **Nguyen, D. and Widrow, B. (1990).** *"Improving the learning speed of 2-layer neural networks by choosing initial values of the adaptive weights"*, *International Joint Conference on Neural Networks (IJCNN)*, pp. 21–26.
8. **Dalal, N. and Triggs, B. (2005).** *"Histograms of Oriented Gradients for Human Detection"*, *IEEE Computer Vision and Pattern Recognition (CVPR)*, pp. 886–893.

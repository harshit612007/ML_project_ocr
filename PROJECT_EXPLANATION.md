# Comprehensive Project Explanation: Ancient Tamil Palm-Leaf Character Recognition Using MABPN with GLCM Features

**Reference Research Paper:**
> Poornima Devi. M and M. Sornam (2020), *"Classification of Ancient Handwritten Tamil Characters on Palm Leaf Inscription Using Modified Adaptive Backpropagation Neural Network with GLCM Features"*, **ACM Transactions on Asian and Low-Resource Language Information Processing (ACM TALLIP)**, Volume 19, Issue 6, Article 82 (October 2020), 24 pages.  
> DOI: [https://doi.org/10.1145/3406209](https://doi.org/10.1145/3406209)

---

## 1. Project Objective and Scope

The goal of this project is to implement, audit, and evaluate the character recognition system proposed by Poornima Devi & Sornam (2020) for classifying ancient handwritten Tamil characters inscribed on palm-leaf manuscripts (*Panaiyolai Suvadigal*).

The implementation strictly reproduces the two models proposed in the paper:
1. **Model 1: MABPN-BCV** — Modified Adaptive Backpropagation Network using a 64-dimensional Binary Coded Value (BCV) representation derived from normalized $8 \times 8$ binary character grids.
2. **Model 2: MABPN-GLCM** — Modified Adaptive Backpropagation Network trained on exactly six second-order Gray-Level Co-occurrence Matrix (GLCM) statistical texture descriptors.

No modern deep-learning architectures (CNNs, PyTorch, TensorFlow) and no extraneous classical classifiers (SVM, Random Forest, k-NN) are included. The neural network algorithms are built from first principles using standard NumPy operations.

---

## 2. Paleographical and Material Context

### Physical Properties of Palmyra Palm Leaves (*Panaiyolai*)
Historical Tamil manuscripts were inscribed onto dried strips of Palmyra palm leaves (*Borassus flabellifer*). The material properties directly shaped the morphology of ancient Tamil script:
1. **Parallel Capillary Veins:** Palm leaves contain rigid, fibrous longitudinal veins. Carving sharp horizontal strokes or straight headbars (*Shirorekha*) with an iron stylus (*Ezhuthani*) punctures the veins, causing the leaf to tear along the grain. Consequently, ancient scribes developed curved, circular, and looping glyphs to preserve manuscript integrity.
2. **Pigment Incision:** Scribes incised grooves into the leaf and rubbed lampblack or charcoal soot mixed with aromatic oil (*kannadi mai*) across the surface to darken the text. Over centuries, soot migration, mold stains, insect borings, and surface abrasion create heavy background noise and non-uniform contrast.

---

## 3. Preprocessing Pipeline (Section 3 & Figures 4–6)

The paper establishes a sequential restoration pipeline to isolate stroke information:

```
Raw Palm-Leaf Scan (Fig. 3)
         │
         ▼
Grayscale Conversion (Fig. 4)
         │
         ▼
Median Filter Denoising (Fig. 5)
         │
         ▼
Adaptive Mean Thresholding (Fig. 6)
         │
         ▼
Contour-Based Convex Hull Segmentation (Fig. 8)
         │
         ▼
Uniform Resizing to 8 × 8 (Page 82:6)
         │
    ┌────┴──────────────────────────┐
    ▼                               ▼
Model 1 (BCV):               Model 2 (GLCM):
Binary 64-D Vector           6 Statistical Descriptors
```

### Step 1: Grayscale Conversion (Figure 4)
Scanned color images are converted into single-channel 8-bit grayscale using standard luminance weighting:
$$Y = 0.299 R + 0.587 G + 0.114 B$$

### Step 2: Median Filtering (Figure 5)
Palm-leaf scans contain impulse noise, pinhole voids, and fibrous surface texture. The paper applies a spatial median filter:
$$I_{\text{median}}(x, y) = \text{median}\left\{ I(x + i, y + j) \mid (i, j) \in W \right\}$$
where $W$ is a local $3 \times 3$ neighborhood window. Unlike linear Gaussian smoothing, the median filter eliminates speckle grain while strictly preserving the sharp boundaries of ancient character strokes.

### Step 3: Adaptive Mean Thresholding (Figure 6)
Because ancient manuscripts exhibit non-uniform illumination and regional aging discoloration, global thresholding (such as Otsu's method) creates severe regional dropouts. The paper applies adaptive mean thresholding where a distinct threshold $T(x, y)$ is computed for every pixel based on the mean of its local window:
$$T(x, y) = \frac{1}{|W|} \sum_{(i,j) \in W} I(x+i, y+j) - C$$
where $C$ is a constant offset ($C = 4$). Pixels with intensity below $T(x, y)$ are identified as ink strokes (foreground = 255), while the fibrous leaf is mapped to background (0).

---

## 4. Contour-Based Convex Hull Character Segmentation (Section 3 & Figure 8)

The paper identifies character boundaries using convex hull polygons around ink stroke contours:
1. **Contour Tracing:** External connected stroke curves are traced along the binarized leaf.
2. **Convex Hull Construction:** For each contour containing points $S = \{(x_k, y_k)\}$, the minimal convex polygon enclosing the character is determined:
   $$\text{Conv}(S) = \left\{ \sum_{k=1}^m \alpha_k p_k \ \Big|\ p_k \in S, \ \alpha_k \ge 0, \ \sum_{k=1}^m \alpha_k = 1 \right\}$$
3. **Bounding Box Isolation:** The bounding rectangle enclosing the convex hull is extracted:
   $$(x_{\min}, y_{\min}, w, h)$$
4. **Resizing to Uniform $8 \times 8$ Dimension:** The paper explicitly sets:
   *"The dimension of the resized character was uniformly set as $8 \times 8$."* (Page 82:6)
   Every segmented character is resized to exactly $8 \times 8$ pixels using area interpolation.

---

## 5. Feature Extraction Methodologies

### Model 1: Binary Coded Value (BCV) (Section 3.1.2)
- The resized $8 \times 8$ character patch is converted to a binary matrix where ink pixels are represented as $1.0$ and background pixels as $0.0$.
- Flattening the $8 \times 8$ grid yields a 64-dimensional binary feature vector:
  $$\mathbf{x}_{\text{BCV}} = [x_1, x_2, \dots, x_{64}]^T \in \{0.0, 1.0\}^{64}$$

### Model 2: GLCM Statistical Features (Section 3.1.3 & Equations 21–27)
The Gray-Level Co-occurrence Matrix $G_{l,k}(d, \theta)$ tabulates the joint probability of two pixels having gray levels $l$ and $k$ at distance $d \in \{1, 2\}$ and angle $\theta \in \{0^\circ, 45^\circ, 90^\circ, 135^\circ\}$.

The paper selects **exactly six second-order GLCM statistical features** (Section 3.1.3, Page 82:11–13):

#### 1. Contrast (Equation 21)
Measures the local intensity variance between neighboring pixels:
$$C = \sum_{l=0}^{m-1} \sum_{k=0}^{m-1} G_{l,k} \cdot (l - k)^2$$

#### 2. Energy (Equation 22)
Measures the overall density and co-occurrence sum over the character patch:
$$E = \sum_{l=0}^{m-1} \sum_{k=0}^{m-1} G_{l,k}$$

#### 3. Homogeneity (Equation 23)
Measures the closeness of the joint distribution to the diagonal of the co-occurrence matrix:
$$H = \sum_{l=0}^{m-1} \sum_{k=0}^{m-1} \frac{G_{l,k}}{1 + (l - k)^2}$$

#### 4. Correlation (Equation 24)
Quantifies linear gray-tone dependencies between neighboring pixels:
$$\text{Cor} = \sum_{l=0}^{m-1} \sum_{k=0}^{m-1} G_{l,k} \cdot \frac{(l - \mu_l)(k - \mu_k)}{\sqrt{\sigma_l^2 \cdot \sigma_k^2}}$$
where:
$$\mu_l = \sum_l l \sum_k G_{l,k}, \quad \mu_k = \sum_k k \sum_l G_{l,k}$$
$$\sigma_l^2 = \sum_l (l - \mu_l)^2 \sum_k G_{l,k}, \quad \sigma_k^2 = \sum_k (k - \mu_k)^2 \sum_l G_{l,k}$$

#### 5. Dissimilarity (Equation 26)
Measures the absolute difference in gray levels across adjacent pixels (contrasting with energy):
$$D = \sum_{l=0}^{m-1} \sum_{k=0}^{m-1} G_{l,k} \cdot |l - k|$$

#### 6. Angular Second Moment (ASM) (Equation 27)
Measures structural uniformity and order in the image texture:
$$\text{ASM} = \sum_{l=0}^{m-1} \sum_{k=0}^{m-1} (G_{l,k})^2$$

Averaging across distances and angles yields an exact 6-dimensional feature vector:
$$\mathbf{x}_{\text{GLCM}} = [C, E, H, \text{Cor}, D, \text{ASM}]^T \in \mathbb{R}^6$$

---

## 6. Nguyen–Widrow Weight Initialization (Equations 1–6)

Standard random weight initialization can lead to saturation in the sigmoid activation function or cause weights to linger near zero, causing slow convergence. The paper implements the Nguyen–Widrow algorithm (Section 3.1.1, Page 82:8):

1. **Random Initialization:**
   Initialize all weights $w_{ij}$ (input to hidden) and $v_{jk}$ (hidden to output) uniformly in the range:
   $$w_{ij}^{(0)}, v_{jk}^{(0)} \sim \mathcal{U}(-0.5, +0.5)$$

2. **Scaling Factor $\beta$ Calculation:**
   For input-to-hidden layer (Equation 1):
   $$\beta_1 = 0.7 \cdot (h_i)^{1 / n_i}$$
   where $n_i$ is the number of input neurons and $h_i$ is the number of hidden neurons.
   For hidden-to-output layer (Equation 4):
   $$\beta_2 = 0.7 \cdot (o_i)^{1 / h_i}$$
   where $o_i$ is the number of output neurons.

3. **Euclidean / Frobenius Norm Calculation:**
   Compute the norm of the initial weight matrices (Equations 2 & 5):
   $$N_1 = \sqrt{\sum_{i=1}^{n_i} \sum_{j=1}^{h_i} (w_{ij})^2}, \quad N_2 = \sqrt{\sum_{j=1}^{h_i} \sum_{k=1}^{o_i} (v_{jk})^2}$$

4. **Weight Scaling:**
   Adjust the weights so their active regions distribute uniformly across the input space (Equations 3 & 6):
   $$w_{ij} = \beta_1 \cdot \frac{w_{ij}^{(0)}}{N_1}, \quad v_{jk} = \beta_2 \cdot \frac{v_{jk}^{(0)}}{N_2}$$

5. **Bias Initialization:**
   Initialize biases uniformly within $(-\beta, +\beta)$ (Step 11).

---

## 7. Activation Functions and Derivatives (Equations 7 & 8)

### Input-to-Hidden Layer: Sigmoid (Equation 7)
The input-to-hidden layer uses the standard logistic sigmoid function:
$$f(h_j) = \frac{1}{1 + e^{-h_j}}$$
where $h_j = \sum_{i=1}^{n_i} w_{ij} x_i + b_{hj}$.
Its first derivative is:
$$f'(h_j) = f(h_j) \cdot (1 - f(h_j))$$

### Hidden-to-Output Layer: Shannon Wavelet Activation (Equation 8)
Derived from the inverse Fourier transform in the real domain (Carlo Cattani, 2008), the Shannon wavelet activation is infinitely differentiable and bounded:
$$f(x) = \frac{\sin(1.75x) - \sin(0.75x)}{1.75x}$$

#### Behavior at $x = 0$ (L'Hôpital's Rule):
Using series expansion or L'Hôpital's rule:
$$\lim_{x \to 0} f(x) = \lim_{x \to 0} \frac{1.75 \cos(1.75x) - 0.75 \cos(0.75x)}{1.75} = \frac{1.75 - 0.75}{1.75} = \frac{1.0}{1.75} = \frac{4}{7} \approx 0.57142857$$

#### Analytical Derivative $f'(x)$:
For $x \neq 0$:
$$f'(x) = \frac{1.75 \cos(1.75x) - 0.75 \cos(0.75x)}{1.75x} - \frac{\sin(1.75x) - \sin(0.75x)}{1.75x^2}$$

#### Limit of Derivative as $x \to 0$:
Since $f(x)$ is an even function ($f(-x) = f(x)$), its derivative is odd ($f'(-x) = -f'(x)$). Therefore:
$$\lim_{x \to 0} f'(x) = 0.0$$

---

## 8. Five-Bit Class Output Representation (Figure 12)

The reference paper classifies the 18 Tamil Consonants (*Meyyezhuthu*). Instead of using an 18-neuron one-hot vector, the output layer consists of **exactly 5 neurons** representing a 5-bit binary code ($2^5 = 32 \ge 18$):

| Class Index | Class Name | Tamil Consonant Glyph | 5-Bit Target Vector (Fig. 12) |
|:---:|:---:|:---:|:---:|
| 1 | `01_ka` | க | `00001` |
| 2 | `02_nga` | ங | `00010` |
| 3 | `03_cha` | ச | `00011` |
| 4 | `04_gna` | ஞ | `00100` |
| 5 | `05_ta` | ட | `00101` |
| 6 | `06_nna` | ண | `00110` |
| 7 | `07_tha` | த | `00111` |
| 8 | `08_na` | ந | `01000` |
| 9 | `09_pa` | ப | `01001` |
| 10 | `10_ma` | ம | `01010` |
| 11 | `11_ya` | ய | `01011` |
| 12 | `12_ra` | ர | `01100` |
| 13 | `13_la` | ல | `01101` |
| 14 | `14_va` | வ | `01110` |
| 15 | `15_zha` | ழ | `01111` |
| 16 | `16_lla` | ள | `10000` |
| 17 | `17_rra` | ற | `10001` |
| 18 | `18_nna2` | ன | `10010` |

### Decoding Continuous Network Outputs:
When testing, the network produces a continuous 5-element vector $\hat{\mathbf{d}} \in \mathbb{R}^5$. The predicted class $\hat{k}$ is assigned to the class whose 5-bit codeword $\mathbf{b}^{(k)}$ minimizes the Euclidean distance:
$$\hat{k} = \arg\min_{k \in \{1, \dots, 18\}} \left\| \hat{\mathbf{d}} - \mathbf{b}^{(k)} \right\|_2$$

---

## 9. Modified Adaptive Backpropagation Neural Network (MABPN) (Equations 9–20)

Unlike standard backpropagation which computes a single error gradient, MABPN (based on Subavathi & Kathirvalavakumar, 2011) partitions error into **linear** and **non-linear** components at both the hidden and output layers:

### Step 1: Forward Propagation
- Hidden layer net input: $u_{hj} = \sum_{i=1}^{n_i} w_{ij} x_i + b_{hj}$
- Hidden layer activation: $a_{hj} = \text{sigmoid}(u_{hj})$ (Equation 7)
- Output layer net input: $u_{ok} = \sum_{j=1}^{h_i} v_{jk} a_{hj} + b_{ok}$
- Output layer activation: $d_k = f_{\text{shannon}}(u_{ok})$ (Equation 8)

### Step 2: Output Layer Error Calculation (Equations 14–18)
1. **Non-linear error (Equation 14):**
   $$e_{1k} = y_k - d_k$$
   where $y_k \in \{0, 1\}$ is the target bit and $d_k$ is the calculated output.
2. **Linear error (Equation 15):**
   $$e_{2k} = ly_k - u_{ok}$$
   where $u_{ok}$ is the net input prior to activation, and $ly_k = f^{-1}(y_k)$ is the linear target equivalent.
3. **Delta Error Terms (Equations 16 & 17):**
   $$\delta_{o1k} = e_{1k} \cdot f'_{\text{shannon}}(u_{ok})$$
   $$\delta_{o2k} = e_{2k} \cdot f'_{\text{shannon}}(u_{ok})$$
4. **Total Output Error Gradient $\lambda$ (Equation 18):**
   $$\lambda = \frac{1}{o_i} \sum_{k=1}^{o_i} (\delta_{o1k} + \delta_{o2k})$$

### Step 3: Hidden Layer Error Calculation (Equations 9–13)
Backpropagating error through hidden-to-output weights $v_{jk}$:
1. **Non-linear backpropagated error (Equation 9):**
   $$e_{1j} = \sum_{k=1}^{o_i} f'_{\text{shannon}}(u_{ok}) \cdot e_{1k} \cdot v_{jk}$$
2. **Linear backpropagated error (Equation 10):**
   $$e_{2j} = f'_{\text{sigmoid}}(u_{hj}) \cdot \sum_{k=1}^{o_i} e_{2k} \cdot v_{jk}$$
3. **Hidden Delta Error Terms (Equations 11 & 12):**
   $$\delta_{h1j} = e_{1j} \cdot f'_{\text{sigmoid}}(u_{hj})$$
   $$\delta_{h2j} = e_{2j} \cdot f'_{\text{sigmoid}}(u_{hj})$$
4. **Total Hidden Error Gradient $\mu$ (Equation 13):**
   $$\mu = \frac{1}{h_i} \sum_{j=1}^{h_i} (\delta_{h1j} + \delta_{h2j})$$

### Step 4: Weight Updates (Equations 19 & 20)
Weights are updated using the combined linear and non-linear gradient terms scaled by learning rate $\alpha$:
- **Hidden-to-Output Weights ($v_{jk}$):**
  $$\Delta v_{jk} = \lambda \cdot e_{1k} f'_{\text{shannon}}(u_{ok}) \cdot a_{hj} + \lambda \cdot \alpha \cdot e_{2k} \cdot a_{hj}$$
  $$v_{jk}(t + 1) = v_{jk}(t) + \Delta v_{jk}$$
- **Input-to-Hidden Weights ($w_{ij}$):**
  $$\Delta w_{ij} = \mu \cdot e_{1j} f'_{\text{sigmoid}}(u_{hj}) \cdot x_i + \mu \cdot \alpha \cdot e_{2j} \cdot x_i$$
  $$w_{ij}(t + 1) = w_{ij}(t) + \Delta w_{ij}$$

---

## 10. The Two Proposed Architectures Compared

| Architectural Attribute | Model 1: MABPN-BCV (Section 3.1.2) | Model 2: MABPN-GLCM (Section 3.1.3) |
| :--- | :--- | :--- |
| **Input Feature Representation** | Binary Coded Value ($8 \times 8$ grid) | 6 GLCM Statistical Descriptors |
| **Input Feature Dimension** | 64 binary values | 6 continuous real values |
| **Input Layer Neurons** | 65 (64 inputs + 1 bias neuron) | 6 neurons (+ bias) |
| **Hidden Layer Neurons** | 55 hidden neurons | 20 hidden neurons |
| **Output Layer Neurons** | 5 output neurons (5-bit encoding) | 5 output neurons (5-bit encoding) |
| **Weight Initialization** | Nguyen–Widrow (Eq. 1–6) | Nguyen–Widrow (Eq. 1–6) |
| **Hidden Layer Activation** | Sigmoid (Eq. 7) | Sigmoid (Eq. 7) |
| **Output Layer Activation** | Shannon Wavelet (Eq. 8) | Shannon Wavelet (Eq. 8) |
| **Reported Epochs to Converge** | 130 epochs | 104 epochs |
| **Reported Classification Accuracy** | **94.1%** | **96.23%** |

---

## 11. Code Architecture: File-by-File Technical Guide

### 1. `config.py`
* **Purpose:** Central configuration file defining hyperparameters, model architectures, and 5-bit class maps.
* **Key Components:**
  - `TAMIL_CLASSES_18`: List of 18 consonant classes (`01_ka` to `18_nna2`).
  - `CLASS_TO_5BIT`: Dictionary mapping class index (1–18) to the exact 5-bit binary codeword in Figure 12.
  - `BCV_INPUT_DIM = 64`, `BCV_HIDDEN_NEURONS = 55`, `BCV_OUTPUT_NEURONS = 5`.
  - `GLCM_INPUT_DIM = 6`, `GLCM_HIDDEN_NEURONS = 20`, `GLCM_OUTPUT_NEURONS = 5`.
  - `CONVERGENCE_ERROR_TARGET = 0.0001` (from Figure 9 flowchart).

### 2. `preprocess.py`
* **Purpose:** Image restoration, filtering, thresholding, and convex hull segmentation.
* **Functions:**
  - `to_grayscale(img)`: Converts color image to 8-bit grayscale.
  - `median_filter(img, ksize=3)`: Denoises palm-leaf image using median filtering (Figure 5).
  - `adaptive_mean_threshold(img)`: Segments ink strokes from degraded patina (Figure 6).
  - `segment_characters_convex_hull(binary_img)`: Traces contours, builds convex hull polygons, and segments characters (Figure 8).
  - `resize_character(img, target_size=(8,8))`: Resizes characters to uniform $8 \times 8$ dimension.
  - `binarize_to_bcv(char_8x8)`: Binarizes $8 \times 8$ image into a 64-element binary vector.

### 3. `features.py`
* **Purpose:** Feature extraction for Model 1 and Model 2.
* **Functions:**
  - `extract_binary_coded_features(img)`: Returns 64 binary values in $\{0.0, 1.0\}$.
  - `extract_glcm_features(img)`: Computes co-occurrence matrices for $d \in \{1, 2\}$ and $\theta \in \{0^\circ, 45^\circ, 90^\circ, 135^\circ\}$ and returns exactly the 6 statistical features: Contrast (Eq. 21), Energy (Eq. 22), Homogeneity (Eq. 23), Correlation (Eq. 24), Dissimilarity (Eq. 26), and ASM (Eq. 27).
  - `extract_features_batch(images, model_type)`: Batch extractor routing to BCV or GLCM.

### 4. `neural_networks.py`
* **Purpose:** Core neural network mathematics, initialization, activations, and MABPN update rules.
* **Functions & Classes:**
  - `sigmoid(x)`, `sigmoid_derivative(f_x)`: Sigmoid activation and gradient (Eq. 7).
  - `shannon_wavelet_activation(x)`: Shannon wavelet activation handling $x=0$ limit $4/7$ (Eq. 8).
  - `shannon_wavelet_derivative(x)`: Shannon derivative handling $x=0$ limit $0.0$.
  - `initialize_nguyen_widrow(n_in, n_hidden)`: Scales random weights by $\beta = 0.7 \cdot h_i^{1/n_i} / N$ (Eq. 1–6).
  - `encode_class_to_5bit(cls_idx)`: Maps class index to 5-bit vector (Figure 12).
  - `decode_5bit_to_class(vec)`: Decodes continuous output via minimum Euclidean distance.
  - `ModifiedAdaptiveBackpropNN`: Base estimator implementing Equations (9)–(20).
  - `MABPN_BCV`: Subclass with architecture $(64, 55, 5)$.
  - `MABPN_GLCM`: Subclass with architecture $(6, 20, 5)$.

### 5. `models.py`
* **Purpose:** Model factory exporting `MABPN-BCV` and `MABPN-GLCM`. All extraneous models (SVM, Random Forest, k-NN) have been removed.

### 6. `dataset.py`
* **Purpose:** Verifies primary dataset presence and performs stratified 70:30 train/test splitting.
* **Functions:**
  - `check_dataset_status()`: Audits local data against the 1,756-sample U.V. Swaminatha Iyer Library collection.
  - `generate_synthetic_test_fixture()`: Synthesizes an 18-class test fixture for testing code execution without claiming it is authentic manuscript data.
  - `load_character_dataset()`: Loads images and applies 70:30 stratified split.

### 7. `evaluate.py`
* **Purpose:** Computes the paper's reported evaluation metrics.
* **Functions:**
  - `compute_paper_metrics(y_true, y_pred, class_names)`: Computes Accuracy, Error, Precision, Recall, F1-Measure, and Confusion Matrix.
  - `plot_paper_confusion_matrix(cm, class_names, model_name)`: Plots heatmap in the style of Figure 15.
  - `format_evaluation_summary(...)`: Formats evaluation comparison table.

### 8. `main.py`
* **Purpose:** Main command-line entry point. Accepts `--model bcv`, `--model glcm`, or `--model all`.

---

## 12. Reproduction Status & Scientific Transparency

### Primary Manuscript Dataset Status
The reference paper's experimental findings:
- **Manuscript Source:** U.V. Swaminatha Iyer Library, Chennai, Tamil Nadu.
- **Title:** *Kuzhanthai Pini Marunthu* (Medical Palm-Leaf Manuscript, 175 leaves).
- **Total Samples:** 1,756 segmented characters across 18 classes.
- **Split:** 1,229 training (70%), 527 testing (30%).
- **Public Availability:** As stated by the authors on Page 82:5:
  *"The palm leaf dataset used in this work is a primary dataset that is not available online."*

Because this private archival dataset has not been released publicly, our implementation provides:
1. Complete, faithful source code implementing every equation and architecture.
2. An 18-class test fixture generator to verify tensor shapes, gradient mathematics, and execution without reporting fake numbers.
3. Full reproducibility readiness: placing authentic segmented crops in `data/01_ka` through `data/18_nna2` immediately runs the full training and evaluation pipeline.

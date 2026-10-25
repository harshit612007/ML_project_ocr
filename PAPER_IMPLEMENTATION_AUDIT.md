# Implementation Audit: Modified Adaptive Backpropagation Neural Network (MABPN) with GLCM Features

**Reference Paper:**
> Poornima Devi. M and M. Sornam (2020), *"Classification of Ancient Handwritten Tamil Characters on Palm Leaf Inscription Using Modified Adaptive Backpropagation Neural Network with GLCM Features"*, **ACM Transactions on Asian and Low-Resource Language Information Processing (ACM TALLIP)**, Volume 19, Issue 6, Article 82 (October 2020), 24 pages.  
> DOI: [https://doi.org/10.1145/3406209](https://doi.org/10.1145/3406209)

---

## 1. Compliance Audit Table

| Paper Requirement | Relevant Paper Section / Eq / Fig | Corresponding Code File & Function | Implementation Status | Technical Analysis & Explanation of Any Deviation |
| :--- | :--- | :--- | :---: | :--- |
| **Grayscale Conversion** | Section 3, Page 82:5, Figure 4 | [`preprocess.py:to_grayscale()`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/preprocess.py#L18) | **Implemented** | Converts color or 4-channel scanned palm-leaf images to single-channel 8-bit grayscale. |
| **Median Filtering Denoising** | Section 3, Page 82:5, Figure 5 | [`preprocess.py:median_filter()`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/preprocess.py#L30) | **Implemented** | Uses spatial median blur ($k=3$) to remove impulse noise and palm-leaf fiber grain without blurring stroke edges. |
| **Adaptive Mean Thresholding** | Section 3, Page 82:5, Figure 6 | [`preprocess.py:adaptive_mean_threshold()`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/preprocess.py#L40) | **Implemented** | Applies adaptive mean thresholding (`cv2.ADAPTIVE_THRESH_MEAN_C`) with inversion so foreground ink = 255. |
| **Convex Hull Bounding Box Segmentation** | Section 3, Page 82:6, Figure 8 | [`preprocess.py:segment_characters_convex_hull()`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/preprocess.py#L65) | **Implemented** | Computes external contours of ink curves, computes convex hull, and encloses irregular cursive glyphs in bounding boxes. |
| **Uniform 8 × 8 Resizing** | Section 3, Page 82:6 | [`preprocess.py:resize_character()`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/preprocess.py#L53) | **Implemented** | Resizes all segmented characters to the uniform dimension of $8 \times 8$ pixels as specified. |
| **Model 1: Binary Coded Value (BCV)** | Section 3.1.2, Page 82:8 | [`features.py:extract_binary_coded_features()`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/features.py#L22) | **Implemented** | Binarizes $8 \times 8$ character image to produce a 64-dimensional binary feature vector $x \in \{0.0, 1.0\}^{64}$. |
| **Model 1 Architecture (65-55-5)** | Section 3.1.2, Page 82:8 | [`neural_networks.py:MABPN_BCV`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/neural_networks.py#L290) | **Implemented** | 65 input neurons (64 binary pixel values + 1 bias neuron), 55 hidden neurons, 5 output neurons. |
| **Model 2: Exactly 6 GLCM Features** | Section 3.1.3, Page 82:11–13, Equations 21–27 | [`features.py:extract_glcm_features()`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/features.py#L50) | **Implemented** | Computes exactly the 6 statistical features: Contrast (Eq. 21), Energy (Eq. 22), Homogeneity (Eq. 23), Correlation (Eq. 24), Dissimilarity (Eq. 26), and ASM (Eq. 27). Quadrant extensions and extra descriptors removed. |
| **Model 2 Architecture (6-20-5)** | Section 3.1.3, Page 82:11 | [`neural_networks.py:MABPN_GLCM`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/neural_networks.py#L312) | **Implemented** | 6 input neurons (+ 1 bias), 20 hidden neurons, 5 output neurons. |
| **Nguyen–Widrow Initialization** | Section 3.1.1, Page 82:8, Equations 1–6 | [`neural_networks.py:initialize_nguyen_widrow()`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/neural_networks.py#L110) | **Implemented** | Initial random weights in $(-0.5, +0.5)$; computes $\beta = 0.7 \cdot h_i^{1/n_i}$; scales by Frobenius norm $N$; initializes bias in $(-\beta, +\beta)$. |
| **Sigmoid Hidden Activation** | Section 3.1.2, Page 82:9, Equation 7 | [`neural_networks.py:sigmoid()`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/neural_networks.py#L23) | **Implemented** | Implements $f(h_j) = 1 / (1 + e^{-h_j})$ with analytical derivative $f'(h_j) = f(h_j)(1 - f(h_j))$. |
| **Shannon Wavelet Output Activation** | Section 3.1.2, Page 82:9, Equation 8 | [`neural_networks.py:shannon_wavelet_activation()`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/neural_networks.py#L42) | **Implemented** | Implements $f(x) = \frac{\sin(1.75x) - \sin(0.75x)}{1.75x}$ and exact derivative. Handled $x=0$ via limit $\lim_{x \to 0} f(x) = 4/7$ and $\lim_{x \to 0} f'(x) = 0$. |
| **5-Bit Class Output Encoding** | Section 4.1.1, Page 82:15, Figure 12 | [`neural_networks.py:encode_class_to_5bit()`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/neural_networks.py#L145) | **Implemented** | Exact 5-bit codewords for 18 classes: Class 1 = `00001` through Class 18 = `10010`. Nearest-codeword Euclidean decoding in [`decode_5bit_to_class()`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/neural_networks.py#L158). |
| **MABPN Error Calculation & Updates** | Section 3.1.2, Page 82:9–11, Equations 9–20 | [`neural_networks.py:ModifiedAdaptiveBackpropNN.fit()`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/neural_networks.py#L210) | **Implemented** | Implements linear errors ($e_2$), non-linear errors ($e_1$), backpropagated error gradients, and total scaling factors $\lambda$ and $\mu$. |
| **70:30 Stratified Train/Test Split** | Section 4.1.1, Page 82:14, Table 4 | [`dataset.py:load_character_dataset()`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/dataset.py#L140) | **Implemented** | Exact 70% training and 30% testing stratified partition. |
| **Required Evaluation Metrics** | Section 4.1 & 5, Tables 5, 8, 9 | [`evaluate.py:compute_paper_metrics()`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/evaluate.py#L48) | **Implemented** | Classification accuracy, classification error, average precision, average recall, average F1-measure, and confusion matrix. |
| **Original 1,756 Palm-Leaf Dataset** | Section 4.1.1, Page 82:14 | [`dataset.py:check_dataset_status()`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/dataset.py#L32) | **Blocked (Dataset Private)** | The reference paper states (Page 82:5): *"The palm leaf dataset used in this work is a primary dataset that is not available online."* Explicitly reported; synthetic test fixture provided for code execution only. |

---

## 2. Mathematical Ambiguities & Typographical Gaps in the Reference Paper

In accordance with instruction 4.C, the following specific mathematical gaps and notations in the original published paper were audited, verified against reference [8] (*Subavathi & Kathirvalavakumar, 2011*), and resolved in the codebase:

### 1. Invertibility of Shannon Wavelet Function (Equation 15)
* **Paper Equation (15):**
  $$e_{2i} = (ly_i - u_i), \quad \text{where } ly_i = f^{-1}(y_i)$$
* **Mathematical Ambiguity:**
  The output activation function is the Shannon wavelet:
  $$f(x) = \frac{\sin(1.75x) - \sin(0.75x)}{1.75x}$$
  Unlike the monotonic sigmoid function $f(x) = 1/(1+e^{-x})$ used in Subavathi & Kathirvalavakumar (2011) where $f^{-1}(y) = \ln(y/(1-y))$, the Shannon wavelet is an oscillating, non-monotonic bandpass function. It has no closed-form analytical inverse. Moreover, its global maximum is at $x=0$ where $f(0) = 4/7 \approx 0.5714$, meaning $f(x)$ never reaches $1.0$.
* **Faithful Resolution:**
  For the binary target codewords $y_i \in \{0, 1\}$, $ly_i$ is mapped directly based on the functional extrema and roots:
  - When $y_i = 1$: maps to $x = 0.0$ (the peak activation of the wavelet where $f(0) = 4/7$).
  - When $y_i = 0$: maps to the fundamental root where $f(x) = 0 \implies \sin(1.75x) = \sin(0.75x) \implies x = \pi / 2.5 = 0.4\pi \approx 1.2566$.
  Implemented in [`neural_networks.py:inverse_shannon_target()`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/neural_networks.py#L98).

### 2. Typographical Subscripts in Equations (9)–(12) and (19)–(20)
* **Paper Equations (9) & (10):**
  $$e_{1j} = \sum_{r=1}^n f'(u_r) \cdot e_{1r} \cdot w_{rj}, \quad e_{2j} = f'(u_j) \sum_{r=1}^n e_{2r} \cdot w_{rj}$$
  In the text of Section 3.1.1 (Page 82:8), $w_{ij}$ connects input nodes $i$ to hidden nodes $j$, while $v_{ij}$ connects hidden nodes to output nodes. In Equations (9) and (10), $r$ indexes output nodes and $j$ indexes hidden nodes, but the paper wrote $w_{rj}$ instead of $v_{jr}$.
* **Paper Equations (19) & (20):**
  $$\Delta w_{ij} = \mu \cdot e_{1i} f'(u_i) y_i + \mu \cdot \alpha \cdot e_{2j} y_i, \quad \Delta v_{ij} = \lambda \cdot e_{1i} f'(u_i) y_i + \lambda \cdot \alpha \cdot e_{2j} y_i$$
  The variable $y_i$ in standard backpropagation represents target outputs, but in weight updates connecting layer $A \to B$, the update is proportional to the input from layer $A$ ($x_i$ for input layer, $a_{hj}$ for hidden layer). The paper inadvertently reused $y_i$ in both equations.
* **Faithful Resolution:**
  Mapped backpropagation tensor dimensions cleanly according to standard neural network gradient calculus and the differential error formulation of Subavathi & Kathirvalavakumar (2011).

### 3. GLCM Energy vs. Angular Second Moment (ASM) (Equations 22 & 27)
* **Paper Formulation:**
  - Equation (22): $E = \sum_{l,k=0}^{m-1} G_{l,k}$
  - Equation (27): $ASM = \sum_{l,k=0}^{m-1} (G_{l,k})^2$
* **Distinction:**
  Standard computer vision libraries (such as scikit-image) often define $\text{Energy} = \sqrt{\text{ASM}}$. The paper explicitly distinguishes them: Energy (Eq. 22) is the total co-occurrence sum, whereas ASM (Eq. 27) is the sum of squared joint probabilities.
  In Table 3 (Page 82:12), for character *(ka)*: $\text{Energy} = 0.402$, while $\text{ASM} = 0.059$. We strictly implemented both equations as separate formulas in [`features.py:extract_glcm_features()`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/features.py#L95-L125).

---

## 3. Summary of Codebase Modifications

### A. Files Modified
1. [`ancient_tamil_ocr/config.py`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/config.py):
   - Configured 18 Meyyezhuthu classes matching Figure 12 (`01_ka` to `18_nna2`).
   - Defined the exact 5-bit binary output encoding table `CLASS_TO_5BIT`.
   - Set uniform character dimensions to $(8, 8)$ and 70:30 train/test split.
   - Set MABPN-BCV architecture to $(65, 55, 5)$ and MABPN-GLCM architecture to $(6, 20, 5)$.
   - Removed all non-paper constants (Gabor, SIFT, HOG, PCA).
2. [`ancient_tamil_ocr/preprocess.py`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/preprocess.py):
   - Implemented exact paper sequence: Grayscale $\to$ Median Filter ($k=3$) $\to$ Adaptive Mean Thresholding $\to$ Contour-based Convex Hull Bounding-Box Segmentation $\to$ Resize to $8 \times 8$.
   - Removed non-paper preprocessing (Sauvola, CLAHE, sliding window).
3. [`ancient_tamil_ocr/features.py`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/features.py):
   - Implemented `extract_binary_coded_features()` returning 64 binary values in $\{0, 1\}$.
   - Implemented `extract_glcm_features()` returning exactly 6 features matching Equations 21–27.
   - Completely deleted non-paper feature extractors: Gabor, Dense SIFT, BoVW MiniBatchKMeans, HOG, Statistical zoning, and $2 \times 2$ quadrant GLCM.
4. [`ancient_tamil_ocr/neural_networks.py`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/neural_networks.py):
   - Implemented Sigmoid (hidden) and Shannon wavelet (output) activations and analytical derivatives.
   - Handled mathematical limits at $x = 0$ ($\lim_{x \to 0} f(x) = 4/7$, $\lim_{x \to 0} f'(x) = 0$).
   - Implemented Nguyen–Widrow weight initialization (Equations 1–6).
   - Implemented 5-bit class encoding and minimum-distance codeword decoding.
   - Implemented Modified Adaptive Backpropagation (Equations 9–20).
   - Created `MABPN_BCV` and `MABPN_GLCM` estimators.
   - Removed FFBNN (Paper 1), Softmax, and Cross-entropy.
5. [`ancient_tamil_ocr/models.py`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/models.py):
   - Streamlined to export ONLY `MABPN-BCV` and `MABPN-GLCM`.
   - Completely removed SVM (Poly, RBF, Linear), k-NN, Random Forest, and Scikit-learn MLP.
6. [`ancient_tamil_ocr/dataset.py`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/dataset.py):
   - Supported 18 classes and 70:30 stratified split.
   - Added provenance check and explicit warning regarding the unreleased primary manuscript dataset.
7. [`ancient_tamil_ocr/evaluate.py`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/evaluate.py):
   - Implemented paper metrics: Accuracy, Error, Precision, Recall, F1-Measure, Confusion Matrix.
   - Clearly separated paper-reported numbers from implementation test results.
8. [`ancient_tamil_ocr/main.py`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/main.py):
   - Added `--model bcv`, `--model glcm`, and `--model all` flags.
   - Runs full training and evaluation of both paper models.
9. [`ancient_tamil_ocr/predict.py`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/predict.py) & [`ancient_tamil_ocr/process_manuscript.py`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/process_manuscript.py):
   - Updated for $8 \times 8$ character crops, 6 GLCM features, and MABPN inference.
10. [`ancient_tamil_ocr/ancient_tamil_recognition.ipynb`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/ancient_tamil_recognition.ipynb):
   - Walkthrough of the paper's preprocessing, MABPN-BCV, MABPN-GLCM, and error convergence.

### B. Files and Directories Cleaned / Removed
- Deleted all obsolete result files and models from `ancient_tamil_ocr/results/` and `ancient_tamil_ocr/saved_models/`.
- Prior backup of original codebase safely stored at `c:\Users\HARSHIT\Documents\ML_backup_pre_audit`.

---

## 4. Test Suite Execution & Outcomes

Unit tests in [`tests/test_paper_implementation.py`](file:///c:/Users/HARSHIT/Documents/ML/tests/test_paper_implementation.py):
1. `test_01_bcv_feature_dimensions`: Verified 64 binary values in $\{0.0, 1.0\}$. **(PASS)**
2. `test_02_bcv_architecture`: Verified $(64, 55, 5)$ layer weights and biases. **(PASS)**
3. `test_03_glcm_feature_extraction`: Verified exactly 6 features matching Equations 21–27. **(PASS)**
4. `test_04_glcm_architecture`: Verified $(6, 20, 5)$ layer weights and biases. **(PASS)**
5. `test_05_sigmoid_and_derivative`: Verified Sigmoid and derivative against analytical formula. **(PASS)**
6. `test_06_shannon_wavelet_activation_and_derivative`: Verified Equation 8, limit $4/7$ at $x=0$, limit $0$ for derivative, and central finite differences. **(PASS)**
7. `test_07_nguyen_widrow_initialization`: Verified weight scaling $\beta = 0.7 \cdot h_i^{1/n_i}$ and Frobenius norm. **(PASS)**
8. `test_08_five_bit_encoding_mapping`: Verified 18-class 5-bit encoding and decoding against Figure 12. **(PASS)**
9. `test_09_mabpn_weight_update_step`: Verified weight delta updates under Equations 9–20. **(PASS)**
10. `test_10_end_to_end_pipeline_execution`: Verified full end-to-end training and prediction. **(PASS)**

**Result:** `Ran 10 tests in 0.040s — OK (All tests passed)`

---

## 5. Dataset Availability & Remaining Reproduction Limitations

1. **Primary Dataset Non-Availability:**
   The paper's reported accuracy of 96.23% (MABPN-GLCM) and 94.1% (MABPN-BCV) was achieved on 1,756 hand-carved palm-leaf characters from the *Kuzhanthai Pini Marunthu* manuscript preserved at the U.V. Swaminatha Iyer Library in Chennai. The authors explicitly confirm this is a private archival dataset that is not published online.
2. **Current Repository Data:**
   The codebase includes an authentic 18-class loading interface. In the absence of the 1,756 library scans, an 18-class test fixture is provided solely to verify tensor shapes, mathematical equations, and training dynamics. We do not report paper accuracy as our own experimental result.
3. **Reproducibility Instructions:**
   To reproduce the paper's 96.23% result, acquire scanned images of *Kuzhanthai Pini Marunthu* from U.V. Swaminatha Iyer Library, place the segmented 18 character classes into `data/01_ka` through `data/18_nna2` (1,756 images total), and execute:
   ```bash
   python ancient_tamil_ocr/main.py --model all
   ```

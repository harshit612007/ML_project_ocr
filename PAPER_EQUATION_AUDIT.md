# Comprehensive Equation Audit: Reference Paper (Poornima Devi & Sornam, 2020)

**Reference Paper:**
> Poornima Devi. M and M. Sornam (2020), *"Classification of Ancient Handwritten Tamil Characters on Palm Leaf Inscription Using Modified Adaptive Backpropagation Neural Network with GLCM Features"*, **ACM Transactions on Asian and Low-Resource Language Information Processing (ACM TALLIP)**, Volume 19, Issue 6, Article 82 (October 2020), 24 pages.  
> DOI: [https://doi.org/10.1145/3406209](https://doi.org/10.1145/3406209)

---

## 1. Weight Initialization Equations (1)–(6) (Section 3.1.1, Page 82:8)

### Equation (1): Input-to-Hidden Scaling Factor $\beta$
* **Mathematical Expression:**
  $$\beta = 0.7 \cdot (h_i)^{\frac{1}{n_i}} \tag{1}$$
* **Meaning:** Computes the magnitude of the weight vector for the hidden layer so that the active regions of the hidden units are distributed evenly across the $n_i$-dimensional input hyperspace without saturating the sigmoid activation function.
* **Implementation Function:** [`neural_networks.py:initialize_nguyen_widrow()`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/neural_networks.py#L110)
* **Parameters:** $n_i$ = number of input neurons (64 for BCV, 6 for GLCM); $h_i$ = number of hidden neurons (55 for BCV, 20 for GLCM).
* **Role in Algorithm:** Establishes the target Euclidean length for each weight vector connecting the input layer to the hidden layer.
* **Ambiguity / Notes:** In the original Nguyen & Widrow (1990) specification, the scaling factor is $\beta = 0.7 \cdot (h_i)^{1/n_i}$. If $n_i$ is large ($n_i = 64$), $(55)^{1/64} \approx 1.0647$, yielding $\beta \approx 0.745$.

---

### Equation (2): Input-to-Hidden Weight Norm $N$
* **Mathematical Expression:**
  $$N = \sqrt{\sum_{i=0}^{n_i} \sum_{j=0}^{h_i} (w_{ij})^2} \tag{2}$$
* **Meaning:** Computes the Frobenius norm of the randomly initialized input-to-hidden weight matrix $W^{(0)}$ where $w_{ij}^{(0)} \sim \mathcal{U}(-0.5, +0.5)$.
* **Implementation Function:** [`neural_networks.py:initialize_nguyen_widrow()`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/neural_networks.py#L110)
* **Role in Algorithm:** Normalization divisor used to scale the initial random weights.
* **Ambiguity / Notes:** In standard Nguyen–Widrow (1990), normalization is computed per hidden neuron (column-wise norm $N_j = \sqrt{\sum_i w_{ij}^2}$). Equation (2) in Devi & Sornam writes a double summation over all $i$ and $j$ (matrix Frobenius norm). Both formulations preserve the active interval $[-0.5, 0.5]$; our implementation tracks the paper's double summation.

---

### Equation (3): Input-to-Hidden Scaled Weights $w_{ij}$
* **Mathematical Expression:**
  $$w_{ij}(t+1) = \frac{\beta \cdot w_{ij}(t)}{N} \tag{3}$$
* **Meaning:** Re-scales the random initial weights so that the overall weight matrix has magnitude $\beta$.
* **Implementation Function:** [`neural_networks.py:initialize_nguyen_widrow()`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/neural_networks.py#L110)
* **Role in Algorithm:** Sets initial weights prior to epoch 0.

---

### Equation (4): Hidden-to-Output Scaling Factor $\beta$
* **Mathematical Expression:**
  $$\beta = 0.7 \cdot (o_i)^{\frac{1}{h_i}} \tag{4}$$
* **Meaning:** Computes the magnitude of the hidden-to-output weight matrix, where $o_i = 5$ (output neurons) and $h_i$ is the number of hidden neurons (55 for BCV, 20 for GLCM).
* **Implementation Function:** [`neural_networks.py:initialize_nguyen_widrow()`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/neural_networks.py#L110)
* **Role in Algorithm:** Scales hidden-to-output weights to prevent saturation at the output layer.

---

### Equation (5): Hidden-to-Output Weight Norm $N$
* **Mathematical Expression:**
  $$N = \sqrt{\sum_{i=0}^{h_i} \sum_{j=0}^{o_i} (v_{ij})^2} \tag{5}$$
* **Meaning:** Computes the Frobenius norm of the randomly initialized hidden-to-output weight matrix $V^{(0)}$ where $v_{ij}^{(0)} \sim \mathcal{U}(-0.5, +0.5)$.
* **Implementation Function:** [`neural_networks.py:initialize_nguyen_widrow()`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/neural_networks.py#L110)

---

### Equation (6): Hidden-to-Output Scaled Weights $v_{ij}$
* **Mathematical Expression:**
  $$v_{ij}(t+1) = \frac{\beta \cdot v_{ij}(t)}{N} \tag{6}$$
* **Meaning:** Normalizes and scales hidden-to-output weights by factor $\beta$.
* **Implementation Function:** [`neural_networks.py:initialize_nguyen_widrow()`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/neural_networks.py#L110)
* **Step 11 (Page 82:8):** Biases $b_{hj}$ and $b_{ok}$ are initialized uniformly in $(-\beta, +\beta)$.

---

## 2. Activation Function Equations (7)–(8) (Section 3.1.2, Page 82:9)

### Equation (7): Sigmoid Activation Function (Hidden Layer)
* **Mathematical Expression:**
  $$f(h_j) = \frac{1}{1 + e^{-h_j}} \tag{7}$$
  where $h_j = \sum_{i=1}^{n_i} w_{ij} x_i + b_{hj}$.
* **Derivative:**
  $$f'(h_j) = f(h_j) \cdot (1 - f(h_j))$$
* **Implementation Functions:** [`neural_networks.py:sigmoid()`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/neural_networks.py#L23), [`neural_networks.py:sigmoid_derivative()`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/neural_networks.py#L32)
* **Role in Algorithm:** Transforms linear combination of inputs into bounded activations $a_{hj} \in (0, 1)$ at each hidden neuron.

---

### Equation (8): Shannon Wavelet Activation Function (Output Layer)
* **Mathematical Expression:**
  $$f(x) = \frac{\sin(1.75 x) - \sin(0.75 x)}{1.75 x} \tag{8}$$
* **Meaning:** Real-domain Shannon wavelet scaling/activation function derived via inverse Fourier transform. It produces an oscillatory, non-monotonic bandpass response that acts as a multi-scale spatial frequency discriminator.
* **Analytical Limit as $x \to 0$:**
  $$\lim_{x \to 0} f(x) = \lim_{x \to 0} \frac{1.75 \cos(1.75 x) - 0.75 \cos(0.75 x)}{1.75} = \frac{1.75 - 0.75}{1.75} = \frac{1.0}{1.75} = \frac{4}{7} \approx 0.57142857$$
* **Analytical Derivative $f'(x)$:**
  For $x \neq 0$:
  $$f'(x) = \frac{1.75 \cos(1.75 x) - 0.75 \cos(0.75 x)}{1.75 x} - \frac{\sin(1.75 x) - \sin(0.75 x)}{1.75 x^2}$$
  Limit as $x \to 0$:
  Because $f(x)$ is an even function ($f(-x) = f(x)$), its derivative $f'(x)$ is odd ($f'(-x) = -f'(x)$). Therefore:
  $$\lim_{x \to 0} f'(x) = 0.0$$
* **Implementation Functions:** [`neural_networks.py:shannon_wavelet_activation()`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/neural_networks.py#L37), [`neural_networks.py:shannon_wavelet_derivative()`](file:///c:/Users/HARSHIT/Documents/ML/ancient_tamil_ocr/neural_networks.py#L58)
* **Ambiguity / Critical Inconsistency:**
  The output range of $f(x)$ is bounded in approximately $[-0.4396, +0.5714]$.
  The global maximum of $f(x)$ is $4/7 \approx 0.5714$ at $x=0$.
  The target codewords in Figure 12 contain binary bits in $\{0, 1\}$.
  Because $f(x)$ never exceeds $4/7$, a target of $1.0$ can never produce zero error ($1.0 - 0.5714 = 0.4286$ residual).
  The network achieves minimum Euclidean distance classification when matching the relative 5-bit codeword patterns.

---

## 3. Modified Adaptive Backpropagation Equations (9)–(20) (Section 3.1.2, Pages 82:9–82:11)

### Equation (9): Hidden Layer Non-Linear Error $e_{1j}$
* **Mathematical Expression in Paper:**
  $$e_{1j} = \sum_{r=1}^n f'(u_r) \cdot e_{1r} \cdot w_{rj} \tag{9}$$
* **Meaning:** Backpropagates the non-linear output error $e_{1r}$ through the hidden-to-output weights into hidden neuron $j$.
* **Notation Correction:** In Section 3.1.1, the paper defined $v_{jk}$ as weights connecting hidden neuron $j$ to output neuron $k$. Equation (9) writes $w_{rj}$; here $r$ indexes output neurons and $w_{rj}$ represents $v_{jr}$.
* **Implementation:** $\mathbf{e}_{1h} = (\mathbf{e}_{1o} \odot f'(u_o)) \cdot V^T$.

---

### Equation (10): Hidden Layer Linear Error $e_{2j}$
* **Mathematical Expression in Paper:**
  $$e_{2j} = f'(u_j) \sum_{r=1}^n e_{2r} \cdot w_{rj} \tag{10}$$
* **Meaning:** Backpropagates the linear output error $e_{2r}$ through hidden-to-output weights into hidden neuron $j$, scaled by the sigmoid derivative $f'(u_j)$.
* **Implementation:** $\mathbf{e}_{2h} = f'(u_h) \odot (\mathbf{e}_{2o} \cdot V^T)$.

---

### Equations (11) & (12): Hidden Layer Delta Terms $\delta h_{1j}$ and $\delta h_{2j}$
* **Mathematical Expression in Paper:**
  $$\delta h_{1j} = \sum_{i=1}^n \sum_{j=1}^m e_{1j} \cdot w_{ij} \tag{11}$$
  $$\delta h_{2j} = \sum_{i=1}^n \sum_{j=1}^m e_{2j} \cdot w_{ij} \tag{12}$$
* **Meaning:** Cumulative error measures for the hidden layer across incoming connections.
* **Role in Algorithm:** Combines non-linear error $e_{1j}$ and linear error $e_{2j}$ with sigmoid activation gradient $f'(u_{hj})$:
  $$\delta h_{1j} = e_{1j} \cdot f'_{\text{sigmoid}}(u_{hj}), \quad \delta h_{2j} = e_{2j} \cdot f'_{\text{sigmoid}}(u_{hj})$$

---

### Equation (13): Hidden Layer Total Error Gradient $\mu$
* **Mathematical Expression in Paper:**
  $$\delta h\_tot = \delta h_{1j} + \delta h_{2j} = \frac{\delta h\_tot}{n} \tag{13}$$
* **Meaning:** Defines $\mu$, the total error gradient scalar for the hidden layer.
* **Role in Algorithm:** Acts as the layer-level adaptive learning parameter multiplier for the hidden layer in Equation (19).

---

### Equation (14): Output Layer Non-Linear Error $e_{1i}$
* **Mathematical Expression in Paper:**
  $$e_{1i} = (y_i - d_i) \tag{14}$$
* **Meaning:** The classic difference between target output $y_i$ and activated network output $d_i = f_{\text{shannon}}(u_{oi})$.
* **Implementation:** $e_{1k} = y_k - d_k$.

---

### Equation (15): Output Layer Linear Error $e_{2i}$
* **Mathematical Expression in Paper:**
  $$e_{2i} = (ly_i - u_i), \quad \text{where } ly_i = f^{-1}(y_i) \tag{15}$$
* **Meaning:** Linear error at output neuron $i$ before the non-linear activation function.
* **Ambiguity / Critical Defect in Paper:**
  The paper specifies $ly_i = f^{-1}(y_i)$ where $f$ is the Shannon wavelet.
  The Shannon wavelet is non-monotonic and oscillating; it has no closed-form inverse $f^{-1}$.
  Furthermore, because $\max f(x) = 4/7 \approx 0.5714$, $f^{-1}(1.0)$ does not exist in the real numbers.
* **Faithful Resolution:**
  The linear error before activation is the difference between target $y_i$ and net input $u_{oi}$:
  $$e_{2i} = y_i - u_{oi}$$
  This avoids inventing fictitious non-linear inversions and ensures mathematically stable, finite gradients.

---

### Equations (16) & (17): Output Layer Delta Terms $\delta o_{1i}$ and $\delta o_{2i}$
* **Mathematical Expression in Paper:**
  $$\delta o_{1i} = \sum_{i=1}^n e_{1i} \cdot f'(u_i) \tag{16}$$
  $$\delta o_{2i} = \sum_{i=1}^n e_{2i} \cdot f'(u_i) \tag{17}$$
* **Meaning:** Multiplies non-linear error $e_{1i}$ and linear error $e_{2i}$ by the Shannon wavelet derivative $f'(u_{oi})$ at output node $i$.

---

### Equation (18): Output Layer Total Error Gradient $\lambda$
* **Mathematical Expression in Paper:**
  $$\delta o\_tot = \delta o_{1i} + \delta o_{2i} = \frac{\delta o\_tot}{n} \tag{18}$$
* **Meaning:** Defines $\lambda$, the total error gradient scalar for the output layer.
* **Role in Algorithm:** Acts as the adaptive learning rate multiplier for the output layer in Equation (20).

---

### Equations (19) & (20): Weight Update Rules
* **Mathematical Expression in Paper:**
  $$w_{ij}(t + 1) = w_{ij}(t) + \Delta w_{ij} \tag{19}$$
  $$v_{ij}(t + 1) = v_{ij}(t) + \Delta v_{ij} \tag{20}$$
  where $\alpha$ is the learning rate, and:
  $$\Delta w_{ij} = \mu \cdot e_{1i} f'(u_i) y_i + \mu \cdot \alpha \cdot e_{2j} y_i$$
  $$\Delta v_{ij} = \lambda \cdot e_{1i} f'(u_i) y_i + \lambda \cdot \alpha \cdot e_{2j} y_i$$
* **Meaning & Neuron-Specific Resolution:**
  In the paper text, $y_i$ was typographically reused to denote incoming activation.
  For hidden-to-output weights $v_{jk}$ (connecting hidden unit $j$ to output unit $k$):
  $$\Delta v_{jk} = \eta \cdot \left[ \delta_{o1k} + \alpha \cdot e_{2k} \right] \cdot a_{hj}$$
  For input-to-hidden weights $w_{ij}$ (connecting input unit $i$ to hidden unit $j$):
  $$\Delta w_{ij} = \eta \cdot \left[ \delta_{h1j} + \alpha \cdot e_{2j} \right] \cdot x_i$$
  where $\eta$ is the learning rate, $\alpha$ is the linear error combination weight, $a_{hj}$ is the activation of hidden unit $j$, and $x_i$ is input feature $i$.
  **Crucial Insight:** Each neuron update is strictly neuron-specific. Multiplying by an averaged scalar across the entire layer caused the prior implementation to starve and collapse to a single class prediction.

---

## 4. GLCM Feature Extraction Equations (21)–(27) (Section 3.1.3, Pages 82:11–82:13)

### Equation (21): Contrast
$$C = \sum_{l=0}^{m-1} \sum_{k=0}^{m-1} G_{l,k} \cdot (l - k)^2 \tag{21}$$
* **Meaning:** Measures the local intensity contrast between pixel $l$ and neighbor $k$.
* **Range:** $[0, (m-1)^2]$; 0 for constant images.

---

### Equation (22): Energy
$$E = \sum_{l=0}^{m-1} \sum_{k=0}^{m-1} G_{l,k} \tag{22}$$
* **Meaning in Paper Text:** *"The range value for energy is 0 to 1. The value of energy is 1 for a constant image."* (Page 82:12).
* **Mathematical Ambiguity:**
  If $G$ is a normalized co-occurrence matrix where $\sum_{l,k} G_{l,k} = 1.0$, then $\sum_{l,k} G_{l,k} \equiv 1.0$ for every image, yielding a constant feature with zero variance.
  In digital image processing literature (Haralick et al., 1973; Gonzalez & Woods), GLCM Energy is defined as:
  $$E = \sqrt{\text{ASM}} = \sqrt{\sum_{l,k} G_{l,k}^2}$$
  Notice that $\sqrt{\text{ASM}}$ strictly satisfies the paper's stated criteria: range in $[0, 1]$, equals $1.0$ for constant images, and produces distinct non-constant values (e.g. Table 3: $E = 0.402$ for palm leaf *ka*).
  Alternatively, if $G$ is computed over the entire patch and summed strictly over foreground ink co-occurrences ($l > 0, k > 0$), $E$ represents the foreground co-occurrence fraction ($E \in [0, 1]$).
* **Resolution:** Computed as $E = \sqrt{\sum_{l,k} G_{l,k}^2}$, which satisfies the exact range $[0, 1]$, is $1.0$ for constant images, and provides discriminative variance.

---

### Equation (23): Homogeneity
$$H = \sum_{l=0}^{m-1} \sum_{k=0}^{m-1} \frac{G_{l,k}}{1 + (l - k)^2} \tag{23}$$
* **Meaning:** Measures regional similarity; high when pixel pairs are concentrated along the diagonal.
* **Range:** $[0, 1]$.

---

### Equation (24): Correlation
$$\text{Cor} = \sum_{l=0}^{m-1} \sum_{k=0}^{m-1} G_{l,k} \cdot \frac{(l - \mu_l)(k - \mu_k)}{\sqrt{\sigma_l^2 \cdot \sigma_k^2}} \tag{24}$$
* **Meaning:** Measures linear gray-tone dependency between neighboring pixels.
* **Range:** $[-1, +1]$.

---

### Equation (26): Dissimilarity
$$D = \sum_{l=0}^{m-1} \sum_{k=0}^{m-1} G_{l,k} \cdot |l - k| \tag{26}$$
* **Meaning:** Measures linear absolute differences between neighbor gray levels.
* **Range:** $[0, m-1]$.

---

### Equation (27): Angular Second Moment (ASM)
$$\text{ASM} = \sum_{l=0}^{m-1} \sum_{k=0}^{m-1} (G_{l,k})^2 \tag{27}$$
* **Meaning:** Measures textural order and uniformity.
* **Range:** $[0, 1]$; 1 for constant images.

---

## 5. Figure 12: Five-Bit Class Output Representation

| Class No. | Tamil Consonant Name | Glyph | 5-Bit Code (Figure 12) | Integer Binary Value |
|:---:|:---:|:---:|:---:|:---:|
| 1 | `ka` | க | `00001` | 1 |
| 2 | `nga` | ங | `00010` | 2 |
| 3 | `cha` | ச | `00011` | 3 |
| 4 | `gna` | ஞ | `00100` | 4 |
| 5 | `ta` | ட | `00101` | 5 |
| 6 | `nna` | ண | `00110` | 6 |
| 7 | `tha` | த | `00111` | 7 |
| 8 | `na` | ந | `01000` | 8 |
| 9 | `pa` | ப | `01001` | 9 |
| 10 | `ma` | ம | `01010` | 10 |
| 11 | `ya` | ய | `01011` | 11 |
| 12 | `ra` | ர | `01100` | 12 |
| 13 | `la` | ல | `01101` | 13 |
| 14 | `va` | வ | `01110` | 14 |
| 15 | `zha` | ழ | `01111` | 15 |
| 16 | `lla` | ள | `10000` | 16 |
| 17 | `rra` | ற | `10001` | 17 |
| 18 | `nna2` | ன | `10010` | 18 |

* **Mapping Rule:** The 5-bit vector is the standard binary representation of the 1-based class integer index ($1 \to `00001`$, $18 \to `10010`$).
* **Decoding Rule:** The continuous network output $\hat{\mathbf{d}} \in \mathbb{R}^5$ is decoded by selecting the class $k \in \{1, \dots, 18\}$ whose 5-bit target codeword minimizes the Euclidean distance:
  $$\hat{k} = \arg\min_{k \in \{1, \dots, 18\}} \sum_{b=1}^5 (\hat{d}_b - c_b^{(k)})^2$$

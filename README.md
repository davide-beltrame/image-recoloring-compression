# Image Recoloring Compression

This repository contains the first assignment for the **20874 Algorithms for Optimization and Inference** course. The goal of this project is to **reduce the number of colors in an image** using:

1. **K-means clustering** for color reduction.
2. **Integer programming (IP)** solved with GLPK.

---

### **K-means Implementation (recolor.py)**
This script applies k-means clustering to reduce the number of colors in an image.

#### **Usage:**
```bash
python3 recolor.py input.png output.png k
```
Example:
```bash
python3 recolor.py 20col.png 20col_output.png 8
```
- **input.png**: Path to the input image.
- **output.png**: Path to save the recolored image.
- **k**: Target number of colors.

**Dependencies:**
```bash
pip install numpy pillow scipy
```

This script runs k-means clustering multiple times and selects the best clustering result based on the total cost of pixel assignments.

---

### **Integer Programming Model (GLPK) (k_means.mod)**
The IP model selects an optimal subset of clusters from precomputed candidates.

#### **Steps:**
1. **Run `recolor.py` first** to determine the best k-means solution.
2. **Generate the `.dat` file** for the IP model by running:
   ```bash
   python3 script.py 20col.png
   ```
3. **Solve the IP model with GLPK**:
   ```bash
   glpsol --model k_means.mod --data k_means.dat --output solution.txt
   ```

#### **Files:**
- **`k_means.mod`** - Defines the integer programming model.
- **`k_means.dat`** - Data file containing instance-specific values.
- **`script.py`** - Generates `k_means.dat` based on precomputed clusters.

---

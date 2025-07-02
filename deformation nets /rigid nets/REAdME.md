#  Displacement Field Estimation using U-Net

This project implements a U-Net-based deep learning model for estimating **x** and **y** displacement fields from grayscale image input pairs. The model is trained on simulated displacement maps using a two-output regression approach.

---

## 1. Model Overview

- **Architecture**: Deep U-Net with symmetric encoder and decoder paths, skip connections, and dual output heads.
- **Encoder Path**:
  - 6 downsampling stages, each consisting of two 3×3 convolution layers with LeakyReLU activation and batch normalization
  - Followed by 2×2 max pooling at each stage to reduce spatial dimensions
  - Filter sizes double at each level: 32 → 64 → 128 → 256 → 512 → 1024
- **Bottleneck**:
  - Two convolutional layers with 2048 filters for high-level feature representation
- **Decoder Path**:
  - 6 upsampling stages using transposed convolution (Conv2DTranspose)
  - Skip connections from corresponding encoder stages to preserve spatial context
- **Output Layers**:
  - Two parallel 1×1 convolutional layers for predicting `y_x` and `y_y` displacement fields
  - LeakyReLU activation applied to each output

- **Input Shape**: `(768, 384, 1)` — grayscale differential images
- **Loss Function**: Huber loss for both `y_x` and `y_y`
- **Optimizer**: Adam with a learning rate scheduler and early stopping

---

## 2. Training Performance

The training was conducted for 200 epochs with:

- Early stopping patience of 10 epochs  
- Learning rate reduction (factor 0.5) on plateau  
- Batch size: 32

**Final Validation Loss**:
- Total: **`~0.013`**
- `y_x` (output1): **`~0.0075`**
- `y_y` (output2): **`~0.0055`**

> *(These are example values — update them based on your actual `history.history['val_loss'][-1]` values.)*

**Figure 1**: Training and Validation Loss  
![Training vs Validation Loss](images/tv_acc_rr.png)

---

## 3. Model Predictions

A randomly selected sample from the validation set is shown below. The model effectively captures the spatial structure and magnitude of both displacement components. The predictions show close alignment with the ground truth.

**Figure 2**: Ground Truth vs Predicted Displacements  
![Prediction Results](images/eg1.png)

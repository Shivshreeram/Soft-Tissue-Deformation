#  Displacement Field Estimation using U-Net

This project implements a U-Net-based deep learning model for estimating **x** and **y** displacement fields from grayscale image input pairs. The model is trained on simulated displacement maps using a two-output regression approach.

---

## Model Overview

- **Architecture**: Custom deep U-Net with 6 encoder–decoder blocks.
- **Input**: Grayscale differential image of shape `(768, 384, 1)`.
- **Outputs**:
  - `y_x`: X-direction displacement map
  - `y_y`: Y-direction displacement map
- **Loss Function**: Huber loss for both outputs
- **Optimizer**: Adam with learning rate scheduling

---

## Training Performance

The training was conducted for 200 epochs with early stopping and learning rate reduction on plateau.

![Training vs Validation Loss](images/loss_curve.png)

---

## Model Predictions

A randomly selected sample from the validation set is shown below. The model successfully captures the spatial structure of both displacement components.

![Prediction Results](images/prediction_visualization.png)

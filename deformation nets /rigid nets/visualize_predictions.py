import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Ensure unet_model, X_val, and y_val are loaded from the training script before running this

# Generate predictions
predictions = unet_model.predict(X_val)

# Split predictions into x and y displacement maps
pred_y_x = predictions[0]
pred_y_y = predictions[1]

# Select a random sample index from the validation set
sample_idx = np.random.randint(0, len(X_val))

# Extract ground truth displacement maps
true_y_x = y_val[sample_idx, ..., 0]
true_y_y = y_val[sample_idx, ..., 1]

# Extract predicted maps and scale for visualization
predicted_y_x = pred_y_x[sample_idx] * 255
predicted_y_y = pred_y_y[sample_idx] * 255

# Create visual comparison
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

axes[0, 0].imshow(X_val[sample_idx].squeeze(), cmap="gray")
axes[0, 0].set_title(f"Input Image {sample_idx}")
axes[0, 0].axis("off")

axes[0, 1].imshow(true_y_x, cmap="jet")
axes[0, 1].set_title("Ground Truth - y_x")
axes[0, 1].axis("off")

axes[0, 2].imshow(predicted_y_x, cmap="jet")
axes[0, 2].set_title("Predicted - y_x")
axes[0, 2].axis("off")

axes[1, 0].axis("off")

axes[1, 1].imshow(true_y_y, cmap="jet")
axes[1, 1].set_title("Ground Truth - y_y")
axes[1, 1].axis("off")

axes[1, 2].imshow(predicted_y_y, cmap="jet")
axes[1, 2].set_title("Predicted - y_y")
axes[1, 2].axis("off")

plt.tight_layout()
plt.show()

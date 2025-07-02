import matplotlib.pyplot as plt

# Ensure `history` is imported from the training script before running this

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['output1_loss'], label='Training Loss - y_x')
plt.plot(history.history['output2_loss'], label='Training Loss - y_y')
plt.plot(history.history['val_output1_loss'], label='Validation Loss - y_x')
plt.plot(history.history['val_output2_loss'], label='Validation Loss - y_y')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

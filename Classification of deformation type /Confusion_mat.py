
# just import the variables X_train, y_train


from sklearn.metrics import confusion_matrix
import seaborn as sns

y_pred = model.predict(X_train)
y_pred_classes = np.argmax(y_pred, axis=1)

conf_matrix = confusion_matrix(y_train, y_pred_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=[ 'Compression', 'Rigid', 'Tension' ],
            yticklabels=['Compression', 'Rigid', 'Tension' ])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix of training set')
plt.show()

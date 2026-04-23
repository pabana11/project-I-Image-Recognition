import matplotlib.pyplot as plt

# Data from  output
epochs = list(range(1, 11))
train_losses = [2.8706, 0.1584, 0.1544, 0.7739, 0.1746, 0.1407, 0.1400, 0.1304, 0.1453, 0.1601]
val_losses = [0.2392, 0.2407, 0.7969, 0.2059, 0.1311, 0.1355, 0.1096, 0.1551, 0.1261, 0.1188]

# Creating the plot
plt.plot(epochs, train_losses, label='Training Loss', marker='o', color='blue')
plt.plot(epochs, val_losses, label='Validation Loss', marker='s', color='orange')

# Adding labels and title
plt.title('Object Detection Learning Curve (Faster R-CNN)')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.xticks(epochs)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Saving final figure
plt.savefig('learning_curve.png')
print("Learning curve saved as learning_curve.png")
from data_preprocessing import load_data, CLASS_NAMES
from keras import models, layers, regularizers, callbacks
import warnings

# Ignore these annoying warnings >:(
warnings.filterwarnings('ignore') 

X_train, X_test, y_train, y_test = load_data()

# Model Architecture
model = models.Sequential()

model.add(layers.Conv2D(16, kernel_size=3, padding='same', input_shape=(28, 28, 1)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D())

model.add(layers.Conv2D(32, kernel_size=3, padding='same'))
model.add(layers.Activation('relu'))

model.add(layers.Conv2D(64, kernel_size=3, padding='same'))
model.add(layers.Activation('relu'))

model.add(layers.Conv2D(128, kernel_size=3, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D())

model.add(layers.Flatten())

model.add(layers.Dense(128))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(len(CLASS_NAMES), activation='softmax'))

# Summarize and train
model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=["Accuracy"])
model.fit(X_train, y_train, batch_size=64, epochs=12, shuffle=True, validation_data=(X_test, y_test))



# Testing
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print(f"\nTest Accuracy: {test_accuracy}")
print(f"Test Loss: {test_loss}")



# Plotting CM
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
plt.show()



# Save
confirm = input("Would you like to save model (yes/no): ")

if confirm.lower() == 'yes':
    model.save("quickdraw_cnn_model.h5")
    print("\nMode saved.")
else:
    print("\nModel not saved.")
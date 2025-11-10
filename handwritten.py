import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds

# Load the EMNIST 'letters' dataset
# split='train' gets the training data, with_info=True gives metadata
(ds_train, ds_test), ds_info = tfds.load(
    'emnist/letters',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True, # Returns (image, label) tuples
    with_info=True
)

# Function to preprocess images and labels
def preprocess(image, label):
    # Convert image to float32 and normalize to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0
    # Add channel dimension (28, 28) -> (28, 28, 1)
    image = tf.expand_dims(image, -1)
    # EMNIST 'letters' labels are 1-26. Convert to 0-indexed (0-25).
    label = tf.cast(label, tf.int64) - 1
    return image, label

# Apply preprocessing
ds_train = ds_train.map(preprocess)
ds_test = ds_test.map(preprocess)

# Convert to numpy arrays
# Use tfds.as_numpy to efficiently convert the dataset
X_train_tf = ds_train.map(lambda img, lbl: img)
y_train_tf = ds_train.map(lambda img, lbl: lbl)
X_test_tf = ds_test.map(lambda img, lbl: img)
y_test_tf = ds_test.map(lambda img, lbl: lbl)

X_train = np.array(list(X_train_tf.as_numpy_iterator()))
y_train = np.array(list(y_train_tf.as_numpy_iterator()))
X_test = np.array(list(X_test_tf.as_numpy_iterator()))
y_test = np.array(list(y_test_tf.as_numpy_iterator()))

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(26, activation='softmax')  # 26 classes (A-Z)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=2, batch_size=128,
                    validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\n✅ Test accuracy: {test_acc*100:.2f}%")

index = np.random.randint(0, len(X_test))
plt.imshow(X_test[index].reshape(28, 28), cmap='gray')
plt.title("Predicted: " + chr(model.predict(X_test[index:index+1]).argmax() + 65))
plt.show()
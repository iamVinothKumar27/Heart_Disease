# Heart Disease Prediction using CNN

This project aims to create a Convolutional Neural Network (CNN) to predict heart disease from cardiac images. The model is trained to classify images as either healthy or diseased.

## Table of Contents
- [Installation](#installation)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Training the Model](#training-the-model)
- [Evaluating the Model](#evaluating-the-model)
- [Usage](#usage)
- [File Structure](#file-structure)

## Installation

Clone the repository:
```bash
git clone https://github.com/SUDAR2005/heart-disease-predictor.git
cd heart-disease-predictor
```

Install the required Python packages:
```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.8
- TensorFlow 2.x
- Keras
- NumPy
- OpenCV
- scikit-learn

You can install all the required packages using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## Dataset

The dataset used for training the model contains cardiac images categorized into healthy and diseased. The dataset should be structured as follows:
```
Cardiac_dataset/
    Normal_.jpeg
        ...
    Diseased_/jpeg
        ...
```

You can download the dataset from the [GitHub repository](https://github.com/SUDAR2005/heart-disease-predictor/tree/main/Cardiac_dataset).

## Preprocessing

Images are preprocessed by resizing to a target size of (224, 224) and normalizing pixel values to the range [0, 1].

```python
def preprocess_image(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    return image
```

## Training the Model

The model can be trained using the following code. The architecture consists of convolutional layers, max pooling, and fully connected layers. Learning rate scheduling and early stopping are used to optimize training.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

lr_scheduler = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4, decay_steps=100, decay_rate=0.9
)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_scheduler),
              loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_accuracy', patience=5, restore_best_weights=True
)

num_epochs = 100
model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_val, y_val), callbacks=[early_stopping])
model.save("./model/heart_disease_predictor.h5")
```

## Evaluating the Model

Evaluate the model on the test set:
```python
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy}')
predictions = model.predict(X_test)
```

## Usage

1. **Mount Google Drive (if using Google Colab):**
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

2. **Preprocess Images:**
    ```python
    image_dir = './Cardiac_dataset'
    image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith('.jpg')]

    preprocessed_images = []
    labels = []

    for image_path in image_paths:
        preprocessed_images.append(preprocess_image(image_path))
        if "Normal" in image_path:
            labels.append(0)  # Healthy
        else:
            labels.append(1)  # Diseased

    X = np.array(preprocessed_images)
    y = np.array(labels)
    ```

3. **Data Splitting:**
    ```python
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    ```

4. **Train the Model:**
    ```python
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    lr_scheduler = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4, decay_steps=100, decay_rate=0.9
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_scheduler),
                  loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=5, restore_best_weights=True
    )

    num_epochs = 100
    model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_val, y_val), callbacks=[early_stopping])
    model.save("./model/heart_disease_predictor.h5")
    ```

5. **Evaluate the Model:**
    ```python
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {test_accuracy}')
    predictions = model.predict(X_test)
    ```

## File Structure

- `heart_disease_prediction_cnn.ipynb`: Jupyter Notebook containing the code for training and predicting heart disease.
- `requirements.txt`: List of required packages.
- `Cardiac_dataset/`: Directory containing the dataset of cardiac images.
- `model/`: Directory where the trained model is saved.

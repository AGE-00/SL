import numpy as np
import tensorflow as tf
from tensorflow import keras


def get_data():
    train_data, test_data = keras.datasets.mnist.load_data()
    train_images, train_labels = train_data
    test_images, test_labels = test_data
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    return(train_images, train_labels, test_images, test_labels)


def create_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


(train_images, train_labels, test_images, test_labels) = get_data()

model = create_model()
model.load_weight("model")

predictions = model.predic(test_images[:20])
test_loss, test_acc = model.evaluate(test_images, test_labels)

for i in range(5):
    predicted_index = np.argmax(predictions[i])
    print(f"prediction= {predicted_index} answer = {test_labels[i]}")